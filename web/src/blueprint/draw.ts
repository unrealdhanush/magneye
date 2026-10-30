import type { Anatomy, Part, Shape, Vec3 } from '../types'

/**
 * Orthographic views of the anatomy in its own frame, laid out as a third-angle engineering sheet: the top view
 * above the front, the right side to its right. Each view maps the object's axes to sheet directions (u right,
 * v up) and looks along the remaining axis.
 */
export type ViewName = 'front' | 'top' | 'right'
interface Axis { axis: 0 | 1 | 2; sign: 1 | -1 }
export interface View { name: ViewName; label: string; u: Axis; v: Axis; depth: 0 | 1 | 2 }

export const VIEWS: Record<ViewName, View> = {
  front: { name: 'front', label: 'FRONT', u: { axis: 0, sign: 1 }, v: { axis: 1, sign: 1 }, depth: 2 },
  // Seen from above with the object's front toward the front view below it.
  top: { name: 'top', label: 'TOP', u: { axis: 0, sign: 1 }, v: { axis: 2, sign: -1 }, depth: 1 },
  // Seen from its right with the object's front toward the front view on its left.
  right: { name: 'right', label: 'RIGHT', u: { axis: 2, sign: -1 }, v: { axis: 1, sign: 1 }, depth: 0 },
}

const ROUND: Shape[] = ['cylinder', 'disc', 'ring', 'cone', 'capsule', 'coil', 'gear', 'spiral']
const FLAT: Shape[] = ['disc', 'ring', 'gear', 'spiral']

/** The axis a round shape turns about, as the 3D viewer reads it from size_mm. */
export function roundAxis(shape: Shape, size: Vec3): 0 | 1 | 2 {
  if (FLAT.includes(shape)) return size.indexOf(Math.min(...size)) as 0 | 1 | 2
  const score = size.map((d, i) => {
    const others = size.filter((_, j) => j !== i)
    return Math.abs(d - (others[0] + others[1]) / 2) - Math.abs(others[0] - others[1])
  })
  return score.indexOf(Math.max(...score)) as 0 | 1 | 2
}

/** A box's span in one view: centre (u, v) and size (w, h) in millimetres. */
export function project(view: View, center: Vec3, size: Vec3) {
  return {
    u: center[view.u.axis] * view.u.sign, v: center[view.v.axis] * view.v.sign,
    w: size[view.u.axis], h: size[view.v.axis],
  }
}

/** One drawable outline of a shape in a view, in sheet units (y down). */
export type Glyph =
  | { kind: 'rect'; x: number; y: number; w: number; h: number; r: number }
  | { kind: 'ellipse'; cx: number; cy: number; rx: number; ry: number }
  | { kind: 'path'; d: string }
  | { kind: 'centerline'; x1: number; y1: number; x2: number; y2: number }

/**
 * Engineering outline of a shape: a round part looked at along its axis is a circle (rings get their bore, gears
 * a tooth line, springs their spiral); seen side-on it is its silhouette with a centre line along the axis.
 * `toSheet` maps a view's millimetres (u right, v up) to sheet units.
 */
export function glyphs(view: View, shape: Shape, center: Vec3, size: Vec3, toSheet: (u: number, v: number) => [number, number], k: number): Glyph[] {
  const p = project(view, center, size)
  const [cx, cy] = toSheet(p.u, p.v)
  const w = Math.max(p.w * k, 1), h = Math.max(p.h * k, 1)
  const rect = (r = 0): Glyph => ({ kind: 'rect', x: cx - w / 2, y: cy - h / 2, w, h, r })
  if (shape === 'sphere') return [{ kind: 'ellipse', cx, cy, rx: w / 2, ry: h / 2 }]
  if (!ROUND.includes(shape)) return [rect(shape === 'rounded_box' ? Math.min(w, h) * 0.18 : 0)]

  const axis = roundAxis(shape, size)
  if (axis === view.depth) {
    const out: Glyph[] = [{ kind: 'ellipse', cx, cy, rx: w / 2, ry: h / 2 }]
    if (shape === 'ring') out.push({ kind: 'ellipse', cx, cy, rx: w * 0.32, ry: h * 0.32 })
    if (shape === 'gear') out.push({ kind: 'ellipse', cx, cy, rx: w * 0.42, ry: h * 0.42 }, { kind: 'ellipse', cx, cy, rx: w * 0.1, ry: h * 0.1 })
    if (shape === 'spiral') out.push({ kind: 'path', d: spiral(cx, cy, w / 2, h / 2) })
    if (shape === 'cylinder' || shape === 'disc' || shape === 'cone') out.push(...cross(cx, cy, w, h))
    return out
  }
  // Side-on: which sheet direction the axis runs along.
  const alongU = axis === view.u.axis
  const line: Glyph = alongU
    ? { kind: 'centerline', x1: cx - w / 2 - 6, y1: cy, x2: cx + w / 2 + 6, y2: cy }
    : { kind: 'centerline', x1: cx, y1: cy - h / 2 - 6, x2: cx, y2: cy + h / 2 + 6 }
  if (shape === 'capsule') return [rect(Math.min(w, h) / 2), line]
  if (shape === 'cone') {
    const d = alongU
      ? `M${cx - w / 2} ${cy - h / 2} L${cx + w / 2} ${cy} L${cx - w / 2} ${cy + h / 2} Z`
      : `M${cx - w / 2} ${cy + h / 2} L${cx} ${cy - h / 2} L${cx + w / 2} ${cy + h / 2} Z`
    return [{ kind: 'path', d }, line]
  }
  if (shape === 'coil') return [rect(), { kind: 'path', d: zigzag(cx, cy, w, h, alongU) }, line]
  return [rect(), line]
}

function cross(cx: number, cy: number, w: number, h: number): Glyph[] {
  return [
    { kind: 'centerline', x1: cx - w / 2 - 6, y1: cy, x2: cx + w / 2 + 6, y2: cy },
    { kind: 'centerline', x1: cx, y1: cy - h / 2 - 6, x2: cx, y2: cy + h / 2 + 6 },
  ]
}

function spiral(cx: number, cy: number, rx: number, ry: number) {
  const turns = 4, n = turns * 24
  return Array.from({ length: n + 1 }, (_, i) => {
    const t = i / n, a = t * turns * Math.PI * 2, r = 0.12 + 0.82 * t
    return `${i ? 'L' : 'M'}${(cx + Math.cos(a) * rx * r).toFixed(1)} ${(cy + Math.sin(a) * ry * r).toFixed(1)}`
  }).join(' ')
}

function zigzag(cx: number, cy: number, w: number, h: number, alongU: boolean) {
  const n = 8
  return Array.from({ length: n + 1 }, (_, i) => {
    const t = i / n - 0.5, s = i % 2 ? 0.5 : -0.5
    const [x, y] = alongU ? [cx + t * w, cy + s * h] : [cx + s * w, cy + t * h]
    return `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`
  }).join(' ')
}

/**
 * A modeled part's real silhouette in a view: the recipe's projected outline, fitted to the part's size and place
 * the same way the 3D view fits its mesh. Through-holes come out as inner contours.
 */
export function outlineGlyphs(view: View, part: Part, outline: { min: Vec3; max: Vec3; polygons: [number, number][][] },
  toSheet: (u: number, v: number) => [number, number]): Glyph[] {
  const fit = (a: { axis: 0 | 1 | 2; sign: 1 | -1 }, raw: number) => {
    const i = a.axis, span = Math.max(outline.max[i] - outline.min[i], 1e-6), mid = (outline.max[i] + outline.min[i]) / 2
    return a.sign * (part.center_mm[i] + (a.sign * raw - mid) * (part.size_mm[i] / span))
  }
  const d = outline.polygons.map(poly => poly.map(([u, v], k) => {
    const [x, y] = toSheet(fit(view.u, u), fit(view.v, v))
    return `${k ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`
  }).join(' ') + ' Z').join(' ')
  return d ? [{ kind: 'path', d }] : []
}

/** The extent the drawing has to show: the object's stated size, grown to cover any part that pokes out. */
export function extent(anatomy: Anatomy): { min: Vec3; max: Vec3 } {
  const half = anatomy.object.dimensions_mm.map(d => d / 2)
  const min = half.map(h => -h) as Vec3, max = [...half] as Vec3
  for (const p of anatomy.parts) for (let i = 0; i < 3; i++) {
    min[i] = Math.min(min[i], p.center_mm[i] - p.size_mm[i] / 2)
    max[i] = Math.max(max[i], p.center_mm[i] + p.size_mm[i] / 2)
  }
  return { min, max }
}

/** The pieces a part is drawn as: its components when it has them, else the part itself. */
export function pieces(part: Part): { shape: Shape; center: Vec3; size: Vec3 }[] {
  const comps = (part.components ?? []).filter(c => c.size_mm?.length === 3 && c.offset_mm?.length === 3)
  if (!comps.length) return [{ shape: part.shape, center: part.center_mm, size: part.size_mm }]
  return comps.map(c => ({ shape: c.shape, center: part.center_mm.map((v, i) => v + c.offset_mm[i]) as Vec3, size: c.size_mm }))
}
