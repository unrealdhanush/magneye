import { useCallback, useEffect, useRef, useState, type MutableRefObject } from 'react'
import { api, HttpError } from '../api'
import type { Anatomy, Part, Vec3 } from '../types'

export type SaveState = 'idle' | 'saving' | 'saved' | 'error' | 'conflict'
const SAVE_DELAY_MS = 500
// Edits to the same field within this window (typing, dragging) undo as one step.
const COALESCE_MS = 1500
const HISTORY = 50

/** The anatomy without one part and everything that pointed at it (as the server's sanitize does). */
function without(a: Anatomy, id: string): Anatomy {
  const failures = a.failures.filter(f => f.culprit_part_id !== id).map(f => ({ ...f, access_path: f.access_path.filter(p => p !== id) }))
  const scenarios = a.scenarios.map(s => ({ ...s, steps: s.steps.filter(st => st.part_id !== id) })).filter(s => s.steps.length)
  return {
    ...a, parts: a.parts.filter(p => p.id !== id), failures, scenarios,
    connections: a.connections.filter(c => c.from !== id && c.to !== id),
  }
}

function freshId(a: Anatomy, base: string) {
  const ids = new Set(a.parts.map(p => p.id))
  let n = 1
  while (ids.has(`${base}_${n}`)) n++
  return `${base}_${n}`
}

/**
 * Editing the anatomy from the blueprint: undo/redo snapshots, and a short-delay save to the server that can be
 * flushed (before the guide reads the anatomy). Guide edits arrive as whole anatomies and can be undone too.
 */
export function useAnatomyEditor(sid: string, latest: MutableRefObject<{ anatomy: Anatomy | null }>, setAnatomy: (a: Anatomy) => void) {
  const history = useRef<{ past: Anatomy[]; future: Anatomy[]; key: string | null; at: number }>({ past: [], future: [], key: null, at: 0 })
  const [, rerender] = useState(0)
  const [saving, setSaving] = useState<SaveState>('idle')
  const pending = useRef<{ anatomy: Anatomy; timer: ReturnType<typeof setTimeout> } | null>(null)

  // The server's revision this tab last saw. Undo restores snapshots with older revisions, so it isn't read off them.
  const serverRevision = useRef(0)
  const send = useCallback(async (a: Anatomy) => {
    setSaving('saving')
    const base = Math.max(serverRevision.current, latest.current.anatomy?.revision ?? 0)
    try {
      const saved = await api.saveAnatomy(sid, a, base)
      serverRevision.current = saved.revision ?? base + 1
      // The server fits the parts together around the edit (neighbours give way, holders take grooves); show that
      // unless another edit is already waiting to be saved. It isn't an undo step: undoing the edit refits again.
      if (!pending.current && JSON.stringify(saved.parts) !== JSON.stringify(a.parts)) {
        latest.current.anatomy = saved
        setAnatomy(saved)
      }
      setSaving('saved')
    } catch (e) {
      if (!(e instanceof HttpError && e.status === 409)) { setSaving('error'); return }
      // Someone saved a newer version (another tab, another person): show theirs rather than overwrite it.
      try {
        const fresh = (await api.session(sid)).anatomy
        if (fresh) { serverRevision.current = fresh.revision ?? 0; latest.current.anatomy = fresh; setAnatomy(fresh) }
      } catch { /* keep what's on screen */ }
      setSaving('conflict')
    }
  }, [sid, latest, setAnatomy])
  const flush = useCallback(async () => {
    const p = pending.current
    if (!p) return
    clearTimeout(p.timer)
    pending.current = null
    await send(p.anatomy)
  }, [send])
  useEffect(() => () => { void flush() }, [flush])

  const apply = useCallback((next: Anatomy, persist = true) => {
    latest.current.anatomy = next
    setAnatomy(next)
    rerender(n => n + 1)
    if (!persist) return
    if (pending.current) clearTimeout(pending.current.timer)
    setSaving('saving')
    pending.current = { anatomy: next, timer: setTimeout(() => void flush(), SAVE_DELAY_MS) }
  }, [latest, setAnatomy, flush])

  const edit = useCallback((change: (a: Anatomy) => Anatomy, key?: string) => {
    const prev = latest.current.anatomy
    if (!prev) return
    const h = history.current, now = performance.now()
    if (!(key && key === h.key && now - h.at < COALESCE_MS)) h.past = [...h.past.slice(-(HISTORY - 1)), prev]
    h.future = []
    h.key = key ?? null
    h.at = now
    apply(change(prev))
  }, [latest, apply])

  /** A new anatomy from elsewhere (the guide changed the model): already saved, but undoable. */
  const recordExternal = useCallback((prev: Anatomy) => {
    const h = history.current
    h.past = [...h.past.slice(-(HISTORY - 1)), prev]
    h.future = []
    h.key = null
    rerender(n => n + 1)
  }, [])

  const undo = useCallback(() => {
    const h = history.current, prev = h.past.pop(), cur = latest.current.anatomy
    if (!prev || !cur) return
    h.future.push(cur)
    h.key = null
    apply(prev)
  }, [latest, apply])
  const redo = useCallback(() => {
    const h = history.current, next = h.future.pop(), cur = latest.current.anatomy
    if (!next || !cur) return
    h.past.push(cur)
    h.key = null
    apply(next)
  }, [latest, apply])

  const editPart = useCallback((id: string, patch: Partial<Part>, key?: string) =>
    edit(a => ({ ...a, parts: a.parts.map(p => (p.id === id ? { ...p, ...patch, edited: true } : p)) }), key ?? `${id}:${Object.keys(patch).join(',')}`), [edit])
  const movePart = useCallback((id: string, center: Vec3, key: string) => editPart(id, { center_mm: center }, key), [editPart])

  const addPart = useCallback((near: Part | null): string | null => {
    const a = latest.current.anatomy
    if (!a) return null
    const id = freshId(a, 'part')
    const part: Part = {
      id, name: 'New part', kind: 'internal', function: '', material: 'plastic', shape: 'box', size_mm: [10, 10, 10],
      center_mm: near ? [near.center_mm[0] + 8, near.center_mm[1] + 8, near.center_mm[2]] : [0, 0, 0], rotation_deg: [0, 0, 0],
      layer: Math.max(1, ...a.parts.map(p => p.layer)), sam_prompt: null, confidence: 0.5, basis: 'inferred', edited: true,
    }
    edit(x => ({ ...x, parts: [...x.parts, part] }))
    return id
  }, [latest, edit])
  const duplicatePart = useCallback((id: string): string | null => {
    const a = latest.current.anatomy, src = a?.parts.find(p => p.id === id)
    if (!a || !src) return null
    const copy: Part = {
      ...src, id: freshId(a, src.id), name: `${src.name} copy`, grounded: false, sam_prompt: null, edited: true,
      center_mm: [src.center_mm[0] + 6, src.center_mm[1] + 6, src.center_mm[2]],
    }
    edit(x => ({ ...x, parts: [...x.parts, copy] }))
    return copy.id
  }, [latest, edit])
  const deletePart = useCallback((id: string) => edit(a => without(a, id)), [edit])

  const h = history.current
  return {
    editPart, movePart, addPart, duplicatePart, deletePart, undo, redo, flush, recordExternal, saving,
    canUndo: h.past.length > 0, canRedo: h.future.length > 0,
  }
}
