import { useEffect, useState } from 'react'
import type { Material, Part, Shape, Vec3 } from '../types'
import { IconX } from '../ui/icons'
import { MATERIAL_LOOK } from '../viewer/palette'

const SHAPES: Shape[] = ['box', 'rounded_box', 'plate', 'cylinder', 'disc', 'ring', 'sphere', 'capsule', 'cone', 'coil', 'gear', 'spiral']
const KINDS: Part['kind'][] = ['exterior', 'internal', 'accessory']
const AXES = ['X', 'Y', 'Z'] as const
const label = (s: string) => s.replace('_', ' ').replace(/^./, c => c.toUpperCase())

/** Edits one field: `key` lets a burst of edits to the same field (typing, dragging) undo as one step. */
export type EditPart = (id: string, patch: Partial<Part>, key?: string) => void

/** A text field that commits when it loses focus or on Enter, so each keystroke isn't an edit. */
function TextField({ value, onCommit, multiline, id }: { value: string; onCommit: (v: string) => void; multiline?: boolean; id: string }) {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])
  const commit = () => { if (draft.trim() && draft !== value) onCommit(draft.trim()) }
  const cls = 'w-full rounded-sm bg-card px-1 py-1 text-callout text-ink shadow-ring focus:shadow-ring-hover'
  return multiline
    ? <textarea id={id} value={draft} rows={2} onChange={e => setDraft(e.target.value)} onBlur={commit} className={`${cls} resize-none`} />
    : <input id={id} value={draft} onChange={e => setDraft(e.target.value)} onBlur={commit}
        onKeyDown={e => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }} className={`${cls} min-h-tap`} />
}

function VecField({ name, value, onChange, min }: { name: string; value: Vec3; onChange: (v: Vec3) => void; min?: number }) {
  return (
    <div className="grid grid-cols-3 gap-1">
      {AXES.map((ax, i) => (
        <label key={ax} className="flex min-h-tap items-center gap-px rounded-sm bg-card px-1 shadow-ring focus-within:shadow-ring-hover">
          <span className="font-mono text-caption text-muted">{ax}</span>
          <input type="number" step={0.5} min={min} value={Number(value[i].toFixed(1))}
            onChange={e => {
              const n = Number(e.target.value)
              if (!Number.isFinite(n) || (min !== undefined && n < min)) return
              const next = [...value] as Vec3
              next[i] = n
              onChange(next)
            }}
            className="tabular w-full min-w-0 bg-transparent text-callout text-ink outline-none" aria-label={`${name} ${ax}`} />
        </label>
      ))}
    </div>
  )
}

/** Edit card for the selected part in the Blueprint space. */
export function Inspector({ part, onEdit, onDuplicate, onDelete, onClose }:
  { part: Part; onEdit: EditPart; onDuplicate: () => void; onDelete: () => void; onClose: () => void }) {
  const field = 'space-y-px'
  const caption = 'block font-mono text-caption uppercase tracking-wider text-muted'
  return (
    <section aria-labelledby="inspector-title" className="glass edge w-[320px] max-w-full space-y-1.5 rounded-xl p-2 text-ink">
      <div className="flex items-start gap-1">
        <div className="min-w-0 flex-1">
          <p className={caption}>{part.edited ? 'Edited by you' : part.grounded ? 'Seen in photo' : label(part.basis ?? 'inferred')}</p>
          <h2 id="inspector-title" className="truncate text-headline">{part.name}</h2>
        </div>
        <button type="button" onClick={onClose} aria-label="Close editor" title="Close"
          className="press grid h-6 w-6 shrink-0 place-items-center rounded-sm hover:bg-ink/5"><IconX className="h-3 w-3" /></button>
      </div>
      <div className={field}>
        <label htmlFor="part-name" className={caption}>Name</label>
        <TextField id="part-name" value={part.name} onCommit={v => onEdit(part.id, { name: v })} />
      </div>
      <div className="grid grid-cols-3 gap-1">
        <label className={field}>
          <span className={caption}>Kind</span>
          <select value={part.kind} onChange={e => onEdit(part.id, { kind: e.target.value as Part['kind'] })}
            className="min-h-tap w-full rounded-sm bg-card px-px text-callout shadow-ring">{KINDS.map(k => <option key={k} value={k}>{label(k)}</option>)}</select>
        </label>
        <label className={field}>
          <span className={caption}>Material</span>
          <select value={part.material} onChange={e => onEdit(part.id, { material: e.target.value as Material })}
            className="min-h-tap w-full rounded-sm bg-card px-px text-callout shadow-ring">
            {(Object.keys(MATERIAL_LOOK) as Material[]).map(m => <option key={m} value={m}>{MATERIAL_LOOK[m].label}</option>)}
          </select>
        </label>
        <label className={field}>
          <span className={caption}>Shape</span>
          <select value={part.shape} onChange={e => onEdit(part.id, { shape: e.target.value as Shape, components: [], model: null })}
            className="min-h-tap w-full rounded-sm bg-card px-px text-callout shadow-ring">{SHAPES.map(s => <option key={s} value={s}>{label(s)}</option>)}</select>
        </label>
      </div>
      <div className={field}>
        <span className={caption}>Size (mm)</span>
        <VecField name="Size" value={part.size_mm} min={0.1} onChange={v => onEdit(part.id, { size_mm: v }, `size:${part.id}`)} />
      </div>
      <div className={field}>
        <span className={caption}>Position (mm)</span>
        <VecField name="Position" value={part.center_mm} onChange={v => onEdit(part.id, { center_mm: v }, `move:${part.id}`)} />
      </div>
      <div className={field}>
        <label htmlFor="part-function" className={caption}>What it does</label>
        <TextField id="part-function" multiline value={part.function} onCommit={v => onEdit(part.id, { function: v })} />
      </div>
      <div className="flex gap-1 pt-px">
        <button type="button" onClick={onDuplicate}
          className="press inline-flex min-h-tap flex-1 items-center justify-center rounded-sm px-1.5 text-subheadline shadow-ring hover:shadow-ring-hover">Duplicate</button>
        <button type="button" onClick={onDelete}
          className="press inline-flex min-h-tap flex-1 items-center justify-center rounded-sm bg-red/10 px-1.5 text-subheadline text-red-text hover:bg-red/15">Delete</button>
      </div>
    </section>
  )
}
