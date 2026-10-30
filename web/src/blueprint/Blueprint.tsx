import { useEffect, useMemo, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from 'react'
import type { Anatomy, Part, Vec3 } from '../types'
import { MATERIAL_LOOK } from '../viewer/palette'
import type { PartOutline } from '../viewer/models'
import { extent, glyphs, outlineGlyphs, pieces, VIEWS, type Glyph, type View, type ViewName } from './draw'

interface Box { x0: number; x1: number; y0: number; y1: number }
/** A sheet layout: the drawing area for the three views, the parts list and the title block, in sheet units. */
export interface Sheet { w: number; h: number; area: Box; list: Box; title: Box }

// Landscape: views on the left, parts list and title block down the right. Portrait: views on top, the list below.
export const LANDSCAPE: Sheet = {
  w: 1600, h: 1000,
  area: { x0: 48, x1: 1000, y0: 64, y1: 952 }, list: { x0: 1032, x1: 1568, y0: 64, y1: 800 }, title: { x0: 1024, x1: 1576, y0: 820, y1: 952 },
}
export const PORTRAIT: Sheet = {
  w: 1000, h: 1600,
  area: { x0: 48, x1: 952, y0: 64, y1: 860 }, list: { x0: 56, x1: 944, y0: 896, y1: 1404 }, title: { x0: 48, x1: 952, y0: 1420, y1: 1552 },
}
const BASIS_LABEL = { seen: 'Seen', documented: 'Documented', inferred: 'Inferred', edited: 'Edited' } as const

const fmt = (mm: number) => (mm < 10 ? mm.toFixed(1) : String(Math.round(mm)))
const delay = (s: number) => ({ '--d': `${s.toFixed(2)}s` }) as CSSProperties

interface Placed { view: View; x0: number; y0: number; w: number; h: number; toSheet: (u: number, v: number) => [number, number] }

/** Where each view sits on the sheet, at one scale `k` (sheet units per millimetre) for all three. */
function layout(anatomy: Anatomy, area: Box) {
  const { min, max } = extent(anatomy)
  const W = max[0] - min[0], H = max[1] - min[1], D = max[2] - min[2]
  const left = area.x0 + 112, gapH = 96, gapV = 88, below = 72
  const k = Math.min((area.x1 - left - gapH - 16) / (W + D), (area.y1 - area.y0 - gapV - below) / (D + H))
  const place = (name: ViewName, x0: number, y0: number): Placed => {
    const view = VIEWS[name]
    const span = (a: { axis: 0 | 1 | 2; sign: 1 | -1 }) => {
      const ends = [min[a.axis] * a.sign, max[a.axis] * a.sign]
      return [Math.min(...ends), Math.max(...ends)]
    }
    const [uLo, uHi] = span(view.u), [vLo, vHi] = span(view.v)
    return { view, x0, y0, w: (uHi - uLo) * k, h: (vHi - vLo) * k, toSheet: (u, v) => [x0 + (u - uLo) * k, y0 + (vHi - v) * k] }
  }
  const top = place('top', left, area.y0)
  const front = place('front', left, area.y0 + D * k + gapV)
  const right = place('right', left + W * k + gapH, front.y0)
  return { k, W, H, D, views: [front, top, right] as const }
}

/** Parts numbered for the parts list: the outside first, then inward by layer. */
function numbered(anatomy: Anatomy): Part[] {
  const rank = { exterior: 0, accessory: 1, internal: 2 } as const
  return [...anatomy.parts].sort((a, b) => a.layer - b.layer || rank[a.kind] - rank[b.kind])
}

/** One outline. Traced-in outlines measure their length as 1 for the draw animation; dashed hidden lines can't. */
function Shape({ g, draw, className, style }: { g: Glyph; draw: boolean; className: string; style?: CSSProperties }) {
  const len = draw ? 1 : undefined
  switch (g.kind) {
    case 'rect': return <rect x={g.x} y={g.y} width={g.w} height={g.h} rx={g.r} pathLength={len} className={className} style={style} />
    case 'ellipse': return <ellipse cx={g.cx} cy={g.cy} rx={g.rx} ry={g.ry} pathLength={len} className={className} style={style} />
    case 'path': return <path d={g.d} pathLength={len} className={className} style={style} />
    case 'centerline': return <line x1={g.x1} y1={g.y1} x2={g.x2} y2={g.y2} className="bp-center bp-fade" style={style} />
  }
}

/**
 * The anatomy as an engineering sheet: front, top and right views (hidden internal parts dashed, round parts with
 * centre lines), overall dimensions, a numbered balloon per part, the parts list and a title block. Hovering or
 * picking a part highlights it in every view and the list; the selection is shared with the 3D space.
 */
export function Blueprint({ anatomy, selectedId, onSelect, onMove, sessionId, outlines, sheet = LANDSCAPE }: {
  anatomy: Anatomy; selectedId: string | null; onSelect: (id: string | null) => void; sessionId: string; sheet?: Sheet
  /** Real silhouettes of modeled parts; other parts are drawn from their shape. */
  outlines?: Map<string, PartOutline>
  /** Drag a selected part in any view to move it; `key` groups one drag into a single undo step. */
  onMove?: (id: string, center: Vec3, key: string) => void
}) {
  const [hovered, setHovered] = useState<string | null>(null)
  const { area: AREA, list: LIST, title: TITLE } = sheet
  const computed = useMemo(() => layout(anatomy, sheet.area), [anatomy, sheet])
  // While dragging, the drawing keeps its scale and placement so the sheet doesn't shift under the pointer.
  const svg = useRef<SVGSVGElement>(null)
  const drag = useRef<{ id: string; view: View; from: [number, number]; start: Vec3; frozen: typeof computed; moved: boolean; key: string } | null>(null)
  const [, setDragging] = useState(false)
  const { k, W, H, D, views } = drag.current?.frozen ?? computed
  const toSheet = (e: { clientX: number; clientY: number }): [number, number] => {
    const el = svg.current!, pt = el.createSVGPoint()
    pt.x = e.clientX; pt.y = e.clientY
    const p = pt.matrixTransform(el.getScreenCTM()!.inverse())
    return [p.x, p.y]
  }
  const startDrag = (e: ReactPointerEvent, p: Part, view: View) => {
    if (!onMove || p.id !== selectedId) return
    e.stopPropagation()
    svg.current?.setPointerCapture(e.pointerId)
    drag.current = { id: p.id, view, from: toSheet(e), start: [...p.center_mm] as Vec3, frozen: computed, moved: false, key: `drag:${p.id}:${Date.now()}` }
    setDragging(true)
  }
  const moveDrag = (e: ReactPointerEvent) => {
    const d = drag.current
    if (!d || !onMove) return
    const [x, y] = toSheet(e)
    const du = (x - d.from[0]) / d.frozen.k, dv = -(y - d.from[1]) / d.frozen.k
    if (!d.moved && Math.hypot(du, dv) * d.frozen.k < 3) return
    d.moved = true
    const next = [...d.start] as Vec3
    next[d.view.u.axis] = Math.round((d.start[d.view.u.axis] + du * d.view.u.sign) * 2) / 2
    next[d.view.v.axis] = Math.round((d.start[d.view.v.axis] + dv * d.view.v.sign) * 2) / 2
    onMove(d.id, next, d.key)
  }
  const endDrag = () => {
    if (!drag.current) return
    // A real drag swallows the click that follows it, so the part stays selected.
    const moved = drag.current.moved
    drag.current = null
    setDragging(false)
    if (moved) suppressClick.current = true
  }
  const suppressClick = useRef(false)
  const parts = useMemo(() => numbered(anatomy), [anatomy])
  const number = new Map(parts.map((p, i) => [p.id, i + 1]))
  const [front, , right] = views
  const state = (id: string) => (id === selectedId ? 'is-selected' : id === hovered ? 'is-hovered' : '')

  // Balloons down the left, ordered top to bottom by where each part sits in the front view, so leaders rarely cross.
  const balloons = useMemo(() => {
    const at = parts.map(p => ({ p, xy: front.toSheet(p.center_mm[0], p.center_mm[1]) })).sort((a, b) => a.xy[1] - b.xy[1])
    const span = AREA.y1 - 40 - AREA.y0, step = Math.min(28, span / Math.max(at.length, 1))
    // Centred on the front view, but shifted to stay whole inside the drawing area.
    const total = step * at.length
    const y0 = Math.min(Math.max(AREA.y0, front.y0 + front.h / 2 - total / 2), AREA.y1 - 40 - total)
    return at.map((b, i) => ({ ...b, by: y0 + i * step + step / 2 }))
  }, [parts, front, AREA])
  const r = Math.min(11, Math.max(7, (balloons[1]?.by ?? 30) - (balloons[0]?.by ?? 0)) / 2.4)

  const rowH = Math.min(26, (LIST.y1 - LIST.y0 - 34) / Math.max(parts.length + 1, 1))
  const small = rowH < 22
  // Columns: number, part, material, size, basis; the part name takes whatever width is left.
  const width = LIST.x1 - LIST.x0
  const cols = [0, 36, width - 300, width - 188, width - 84].map(x => LIST.x0 + x)
  const charW = small ? 6.7 : 7.9
  const nameChars = Math.floor((cols[2] - cols[1] - 10) / charW)
  const cut = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s)
  const titleSplit = TITLE.x0 + (TITLE.x1 - TITLE.x0) * 0.44

  return (
    <svg ref={svg} viewBox={`0 0 ${sheet.w} ${sheet.h}`} className={`blueprint h-full w-full ${drag.current ? 'is-dragging' : ''}`} role="img"
      aria-label={`Blueprint of ${anatomy.object.name}: front, top and right views with ${parts.length} numbered parts`}
      onPointerMove={moveDrag} onPointerUp={endDrag} onPointerCancel={endDrag}
      onClickCapture={e => { if (suppressClick.current) { suppressClick.current = false; e.stopPropagation() } }}
      onClick={() => onSelect(null)}>
      <defs>
        <pattern id="bp-grid" width="16" height="16" patternUnits="userSpaceOnUse"><path d="M16 0H0V16" className="bp-gridline" /></pattern>
        <pattern id="bp-grid-major" width="80" height="80" patternUnits="userSpaceOnUse"><rect width="80" height="80" fill="url(#bp-grid)" /><path d="M80 0H0V80" className="bp-gridline-major" /></pattern>
        <marker id="bp-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M0 1L9 5L0 9Z" className="bp-arrowhead" />
        </marker>
      </defs>
      <rect width={sheet.w} height={sheet.h} className="bp-paper" />
      <rect width={sheet.w} height={sheet.h} fill="url(#bp-grid-major)" />
      <rect x={24} y={24} width={sheet.w - 48} height={sheet.h - 48} pathLength={1} className="bp-frame bp-draw" style={delay(0)} />
      <text x={48} y={50} className="bp-label bp-fade" style={delay(0.2)}>MAGNEYE BLUEPRINT · {anatomy.object.name.toUpperCase()}</text>

      {views.map((pv, vi) => (
        <g key={pv.view.name}>
          <line x1={pv.x0 - 10} y1={pv.y0 + pv.h / 2} x2={pv.x0 + pv.w + 10} y2={pv.y0 + pv.h / 2} className="bp-center bp-fade" style={delay(0.1)} />
          <line x1={pv.x0 + pv.w / 2} y1={pv.y0 - 10} x2={pv.x0 + pv.w / 2} y2={pv.y0 + pv.h + 10} className="bp-center bp-fade" style={delay(0.1)} />
          {/* The selected part is drawn last, so nothing covers it and it can be dragged. */}
          {[...parts.filter(p => p.id !== selectedId), ...parts.filter(p => p.id === selectedId)].map((p, pi) => {
            const hidden = p.kind === 'internal'
            const d = delay(0.15 + vi * 0.15 + Math.min(pi * 0.02, 0.4))
            const real = outlines?.get(p.id)
            const box = real?.views[pv.view.name]?.length
              ? outlineGlyphs(pv.view, p, real.views[pv.view.name], pv.toSheet)
              : pieces(p).flatMap(pc => glyphs(pv.view, pc.shape, pc.center, pc.size, pv.toSheet, k))
            const hit = glyphs(pv.view, 'box', p.center_mm, p.size_mm, pv.toSheet, k)[0] as Extract<Glyph, { kind: 'rect' }>
            return (
              <g key={p.id} className={`bp-part ${hidden ? 'is-hidden' : ''} ${state(p.id)} ${onMove && p.id === selectedId ? 'is-movable' : ''}`}
                onPointerEnter={() => setHovered(p.id)} onPointerLeave={() => setHovered(h => (h === p.id ? null : h))}
                onPointerDown={e => startDrag(e, p, pv.view)}
                onClick={e => { e.stopPropagation(); onSelect(p.id) }}>
                <rect x={hit.x} y={hit.y} width={hit.w} height={hit.h} className="bp-hit" />
                {box.map((g, gi) => <Shape key={gi} g={g} draw={!hidden} className={hidden ? 'bp-line bp-fade' : 'bp-line bp-draw'} style={d} />)}
              </g>
            )
          })}
          <text x={pv.x0} y={pv.y0 + pv.h + (pv.view.name === 'top' ? 28 : 56)} className="bp-label bp-fade" style={delay(0.7 + vi * 0.1)}>
            {pv.view.label} VIEW
          </text>
        </g>
      ))}

      {/* Overall dimensions: width under the front view, height between front and right, depth under the right view. */}
      <g className="bp-fade" style={delay(0.9)}>
        <Dimension x1={front.x0} y1={front.y0 + front.h} x2={front.x0 + front.w} y2={front.y0 + front.h} offset={30} text={fmt(W)} />
        <Dimension x1={front.x0 + front.w} y1={front.y0} x2={front.x0 + front.w} y2={front.y0 + front.h} offset={36} text={fmt(H)} vertical />
        <Dimension x1={right.x0} y1={right.y0 + right.h} x2={right.x0 + right.w} y2={right.y0 + right.h} offset={30} text={fmt(D)} />
      </g>

      {balloons.map(({ p, xy, by }, i) => (
        <g key={p.id} className={`bp-balloon bp-fade ${state(p.id)}`} style={delay(1 + i * 0.015)}
          onPointerEnter={() => setHovered(p.id)} onPointerLeave={() => setHovered(h => (h === p.id ? null : h))}
          onClick={e => { e.stopPropagation(); onSelect(p.id) }}>
          <line x1={AREA.x0 + 16 + r} y1={by} x2={xy[0]} y2={xy[1]} className="bp-leader" />
          <circle cx={xy[0]} cy={xy[1]} r={2} className="bp-dot" />
          <circle cx={AREA.x0 + 16} cy={by} r={r} className="bp-balloon-ring" />
          <text x={AREA.x0 + 16} y={by} className="bp-balloon-text" style={{ fontSize: r * 1.15 }}>{number.get(p.id)}</text>
        </g>
      ))}

      {/* Parts list */}
      <g className="bp-fade" style={delay(0.5)}>
        <text x={LIST.x0} y={LIST.y0 + 4} className="bp-label">PARTS LIST</text>
        <line x1={LIST.x0} y1={LIST.y0 + 14} x2={LIST.x1} y2={LIST.y0 + 14} className="bp-rule" />
        {['No.', 'Part', 'Material', 'Size (mm)', 'Basis'].map((h, i) => (
          <text key={h} x={cols[i]} y={LIST.y0 + 34} className="bp-th">{h}</text>
        ))}
      </g>
      {parts.map((p, i) => {
        const y = LIST.y0 + 34 + (i + 1) * rowH
        const basis = p.edited ? 'edited' : p.grounded ? 'seen' : p.basis ?? 'inferred'
        const cell = `bp-td ${small ? 'is-small' : ''}`
        return (
          <g key={p.id} className={`bp-row bp-fade ${state(p.id)} ${p.edited ? 'is-edited' : ''}`} style={delay(0.6 + i * 0.02)}
            onPointerEnter={() => setHovered(p.id)} onPointerLeave={() => setHovered(h => (h === p.id ? null : h))}
            onClick={e => { e.stopPropagation(); onSelect(p.id) }}>
            <rect x={LIST.x0 - 6} y={y - rowH + 6} width={LIST.x1 - LIST.x0 + 12} height={rowH} className="bp-row-bg" />
            <text x={cols[0]} y={y} className={`${cell} bp-num`}>{number.get(p.id)}</text>
            <text x={cols[1]} y={y} className={cell}>{cut(p.name, nameChars)}</text>
            <text x={cols[2]} y={y} className={cell}>{cut(MATERIAL_LOOK[p.material]?.label ?? p.material, 14)}</text>
            <text x={cols[3]} y={y} className={`${cell} bp-num`}>{p.size_mm.map(fmt).join('×')}</text>
            <text x={cols[4]} y={y} className={cell}>{BASIS_LABEL[basis]}</text>
          </g>
        )
      })}

      {/* Title block */}
      <g className="bp-fade" style={delay(0.3)}>
        <rect x={TITLE.x0} y={TITLE.y0} width={TITLE.x1 - TITLE.x0} height={TITLE.y1 - TITLE.y0} className="bp-titleblock" />
        <line x1={TITLE.x0} y1={TITLE.y0 + 52} x2={TITLE.x1} y2={TITLE.y0 + 52} className="bp-rule" />
        <line x1={titleSplit} y1={TITLE.y0 + 52} x2={titleSplit} y2={TITLE.y1} className="bp-rule" />
        <text x={TITLE.x0 + 12} y={TITLE.y0 + 24} className="bp-title">{cut(anatomy.object.name, Math.floor((TITLE.x1 - TITLE.x0 - 24) / 12.5))}</text>
        <text x={TITLE.x0 + 12} y={TITLE.y0 + 42} className="bp-td">
          {cut([anatomy.object.maker, anatomy.object.model].filter(Boolean).join(' · ') || anatomy.object.category, Math.floor((TITLE.x1 - TITLE.x0 - 24) / 7.9))}
        </text>
        <text x={TITLE.x0 + 12} y={TITLE.y0 + 74} className="bp-th">UNITS</text>
        <text x={TITLE.x0 + 12} y={TITLE.y0 + 92} className="bp-td">Millimetres · scale to fit</text>
        <text x={TITLE.x0 + 12} y={TITLE.y0 + 114} className="bp-th">HIDDEN PARTS</text>
        <text x={TITLE.x0 + 128} y={TITLE.y0 + 114} className="bp-td">dashed</text>
        <text x={titleSplit + 12} y={TITLE.y0 + 74} className="bp-th">DRAWN BY</text>
        <text x={titleSplit + 12} y={TITLE.y0 + 92} className="bp-td">Magneye · Muse Wonder + SAM 3.1</text>
        <text x={titleSplit + 12} y={TITLE.y0 + 114} className="bp-th">SHEET</text>
        <text x={titleSplit + 68} y={TITLE.y0 + 114} className="bp-td bp-num">1 / 1 · {sessionId.slice(0, 8)}</text>
      </g>
    </svg>
  )
}

/** A dimension: extension lines from the feature, an arrowed line `offset` away, and the measurement. */
function Dimension({ x1, y1, x2, y2, offset, text, vertical }:
  { x1: number; y1: number; x2: number; y2: number; offset: number; text: string; vertical?: boolean }) {
  if (vertical) {
    const x = x1 + offset
    return (
      <g>
        <line x1={x1 + 4} y1={y1} x2={x + 6} y2={y1} className="bp-ext" />
        <line x1={x2 + 4} y1={y2} x2={x + 6} y2={y2} className="bp-ext" />
        <line x1={x} y1={y1} x2={x} y2={y2} className="bp-dim" markerStart="url(#bp-arrow)" markerEnd="url(#bp-arrow)" />
        <text x={x - 8} y={(y1 + y2) / 2} className="bp-dimtext" transform={`rotate(-90 ${x - 8} ${(y1 + y2) / 2})`}>{text}</text>
      </g>
    )
  }
  const y = y1 + offset
  return (
    <g>
      <line x1={x1} y1={y1 + 4} x2={x1} y2={y + 6} className="bp-ext" />
      <line x1={x2} y1={y2 + 4} x2={x2} y2={y + 6} className="bp-ext" />
      <line x1={x1} y1={y} x2={x2} y2={y} className="bp-dim" markerStart="url(#bp-arrow)" markerEnd="url(#bp-arrow)" />
      <text x={(x1 + x2) / 2} y={y - 8} className="bp-dimtext">{text}</text>
    </g>
  )
}

/** The sheet in whichever orientation wastes less of the space it is given, re-checked as that space resizes. */
export function FittedBlueprint(props: {
  anatomy: Anatomy; selectedId: string | null; onSelect: (id: string | null) => void; sessionId: string
  onMove?: (id: string, center: Vec3, key: string) => void; outlines?: Map<string, PartOutline>
}) {
  const box = useRef<HTMLDivElement>(null)
  const [portrait, setPortrait] = useState(false)
  useEffect(() => {
    const el = box.current
    if (!el) return
    // Landscape sheets are 1.6:1 and portrait 1:1.6, so 1:1 is where they waste the same.
    const ro = new ResizeObserver(([e]) => setPortrait(e.contentRect.width < e.contentRect.height))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return <div ref={box} className="h-full w-full"><Blueprint {...props} sheet={portrait ? PORTRAIT : LANDSCAPE} /></div>
}
