import { useEffect, useId, useRef, useState, type ReactNode } from 'react'
import type { Space } from '../blueprint/SpaceTabs'
import { blueprintSvg, copyText, download, fileSlug, svgToPng } from '../share'
import type { ViewerHandle } from '../viewer/Viewer'
import { IconDownload, IconLink, IconShare } from './icons'

interface Item { id: string; label: string; hint: string; ready: boolean; run: () => Promise<void>; done: string }

const findSheet = () => document.querySelector<SVGSVGElement>('#space-blueprint svg.blueprint')
const canShare = typeof navigator !== 'undefined' && typeof navigator.share === 'function'

const itemClass = 'press flex w-full min-h-tap items-start gap-1.5 rounded-sm px-1.5 py-1 text-start text-ink hover:bg-ink/5 disabled:opacity-40 disabled:hover:bg-transparent'

/** Share the session's link, or export the current space: the 3D model and view from Reality, the sheet from Blueprint. */
export function ShareMenu({ name, space, ready, getViewer }: {
  name: string; space: Space; ready: boolean; getViewer: () => ViewerHandle | null
}) {
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const [note, setNote] = useState<{ text: string; error: boolean } | null>(null)
  const root = useRef<HTMLDivElement>(null)
  const panelId = useId()
  const button = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: PointerEvent) => { if (!root.current?.contains(e.target as Node)) setOpen(false) }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { setOpen(false); button.current?.focus() } }
    addEventListener('pointerdown', onDown)
    addEventListener('keydown', onKey)
    root.current?.querySelector<HTMLButtonElement>(`[id="${CSS.escape(panelId)}"] button:not(:disabled)`)?.focus()
    return () => { removeEventListener('pointerdown', onDown); removeEventListener('keydown', onKey) }
  }, [open, panelId])

  const run = async (item: { id: string; run: () => Promise<void>; done: string }) => {
    setBusy(item.id); setNote(null)
    try { await item.run(); setNote({ text: item.done, error: false }) }
    catch (e) { setNote({ text: e instanceof Error ? e.message : String(e), error: true }) }
    setBusy(null)
  }

  const link = location.href
  const shareItems: Item[] = [
    { id: 'copy', label: 'Copy link', hint: 'Anyone with the link can open this session.', ready: true, done: 'Link copied.',
      run: async () => { if (!(await copyText(link))) throw new Error('The link could not be copied. Copy it from the address bar.') } },
    ...(canShare ? [{ id: 'share', label: 'Share…', hint: 'Send it with another app.', ready: true, done: '',
      run: async () => { try { await navigator.share({ title: `Magneye: ${name}`, url: link }) } catch (e) { if ((e as Error).name !== 'AbortError') throw e } } }] : []),
  ]

  // Computed while the menu is open, so Blueprint items are ready once the sheet has been opened.
  const exportItems = (): Item[] => {
    if (!open) return []
    const viewer = getViewer(), sheet = findSheet(), slug = fileSlug(name)
    const reality: Item[] = [
      { id: 'glb', label: '3D model (.glb)', hint: 'For Blender, AR and 3D viewers, as currently exploded.', ready: ready && !!viewer, done: 'Downloaded the 3D model.',
        run: async () => download(await viewer!.glb(), `${slug}.glb`) },
      { id: 'view', label: 'Image of this view (.png)', hint: 'The 3D view as you see it.', ready: ready && !!viewer, done: 'Downloaded the image.',
        run: async () => download(await viewer!.png(), `${slug}-3d.png`) },
    ]
    const blueprint: Item[] = [
      { id: 'svg', label: 'Blueprint (.svg)', hint: sheet ? 'Vector sheet for print or editing.' : 'Open Blueprint once to export it.', ready: !!sheet, done: 'Downloaded the blueprint.',
        run: async () => download(new Blob([blueprintSvg(sheet!).svg], { type: 'image/svg+xml' }), `${slug}-blueprint.svg`) },
      { id: 'bpng', label: 'Blueprint (.png)', hint: sheet ? 'High-resolution image.' : 'Open Blueprint once to export it.', ready: !!sheet, done: 'Downloaded the blueprint.',
        run: async () => { const b = blueprintSvg(sheet!); download(await svgToPng(b.svg, b.w, b.h, 2), `${slug}-blueprint.png`) } },
    ]
    return space === 'reality' ? [...reality, ...blueprint] : [...blueprint, ...reality]
  }

  const row = (it: Item, icon: ReactNode) => (
    <button key={it.id} type="button" disabled={!it.ready || busy !== null} onClick={() => run(it)} className={itemClass}>
      <span className="mt-px text-accent-text">{icon}</span>
      <span className="min-w-0">
        <span className="block text-subheadline font-medium">{busy === it.id ? 'Working…' : it.label}</span>
        <span className="block text-footnote text-muted">{it.hint}</span>
      </span>
    </button>
  )

  return (
    <div ref={root} className="relative"
      onBlur={e => { if (open && !e.currentTarget.contains(e.relatedTarget as Node | null)) setOpen(false) }}>
      <button ref={button} type="button" aria-expanded={open} aria-controls={open ? panelId : undefined} onClick={() => { setOpen(o => !o); setNote(null) }}
        title="Share or export" className={`glass press inline-flex min-h-tap min-w-tap items-center justify-center gap-1 rounded-lg px-1.5 text-subheadline ${open ? 'text-accent-text' : 'text-ink'}`}>
        <IconShare className="h-2.5 w-2.5" />
        <span className="sr-only sm:not-sr-only">Share</span>
      </button>
      {open && (
        <div id={panelId} role="group" aria-label="Share or export" className="glass absolute right-0 top-full z-20 mt-1 max-h-[max(280px,calc(60vh-72px))] w-[min(320px,calc(100vw-32px))] overflow-y-auto rounded-lg p-1">
          {shareItems.map(it => row(it, <IconLink className="h-2.5 w-2.5" />))}
          <p className="px-1.5 pb-px pt-1.5 text-footnote font-medium text-muted">Export</p>
          {exportItems().map(it => row(it, <IconDownload className="h-2.5 w-2.5" />))}
          <p role="status" className={`min-h-[18px] px-1.5 pb-1 pt-px text-footnote ${note?.error ? 'text-red-text' : 'text-muted'}`}>{note?.text ?? ''}</p>
        </div>
      )}
    </div>
  )
}
