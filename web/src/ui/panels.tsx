import { useState } from 'react'
import { ThinkingOrb, type OrbState } from 'thinking-orbs'
import { startRecording, type Recorder } from '../audio'
import type { Anatomy, Answer, Failure, Ground, Part, Scenario } from '../types'
import { MATERIAL_LOOK } from '../viewer/palette'
import { IconArrow, IconCheck, IconMic, IconPlay, IconStop, IconWarn } from './icons'

export type StageKey = 'anatomy' | 'ground' | 'shell' | 'answer'
export type StageStatus = 'pending' | 'running' | 'done' | 'error' | 'skipped'
export interface StageState { status: StageStatus; detail?: string }

const STAGES: { key: StageKey; title: string; model: string; orb: OrbState; running: string }[] = [
  { key: 'anatomy', title: 'Identify & infer internals', model: 'Muse Wonder', orb: 'solving', running: 'Reasoning about what’s inside…' },
  { key: 'ground', title: 'Find parts in the photo', model: 'SAM 3.1', orb: 'searching', running: 'Segmenting the object and visible parts…' },
  { key: 'shell', title: 'Reconstruct the shell', model: 'SAM 3D Objects', orb: 'shaping', running: 'Building a 3D mesh from the photo…' },
  { key: 'answer', title: 'Answer your prompt', model: 'Muse Wonder', orb: 'composing', running: 'Writing the answer…' },
]

export function Pipeline({ stages }: { stages: Record<StageKey, StageState> }) {
  return (
    <ol className="space-y-1" aria-label="Pipeline">
      {STAGES.map(s => {
        const st = stages[s.key]
        return (
          <li key={s.key} className="flex items-start gap-1.5 rounded-md bg-card p-1.5 shadow-ring">
            <span className="grid h-3 w-3 shrink-0 place-items-center pt-px">
              {st.status === 'running' ? <ThinkingOrb state={s.orb} size={20} aria-hidden="true" />
                : st.status === 'done' ? <IconCheck className="h-2.5 w-2.5 text-green-text" />
                : st.status === 'error' ? <IconWarn className="h-2.5 w-2.5 text-red-text" />
                : <span className={`h-1 w-1 rounded-full ${st.status === 'skipped' ? 'bg-muted/40' : 'bg-muted'}`} />}
            </span>
            <span className="min-w-0 flex-1">
              <span className="flex items-baseline justify-between gap-1">
                <span className="text-subheadline font-medium text-ink">{s.title}</span>
                <span className="shrink-0 font-mono text-caption2 uppercase tracking-wider text-muted">{s.model}</span>
              </span>
              <span role={st.status === 'running' ? 'status' : undefined}
                className={`block text-footnote ${st.status === 'error' ? 'text-red-text' : 'text-muted'}`}>
                {st.status === 'running' ? s.running : st.detail ?? (st.status === 'pending' ? 'Waiting' : st.status === 'skipped' ? 'Skipped' : '')}
              </span>
            </span>
          </li>
        )
      })}
    </ol>
  )
}

/** The input photo with SAM 3.1's masks traced over it. */
export function PhotoMasks({ imageUrl, ground, anatomy, selectedId, onSelect }:
  { imageUrl: string; ground: Ground | null; anatomy: Anatomy | null; selectedId: string | null; onSelect: (id: string) => void }) {
  const names = new Map(anatomy?.parts.map(p => [p.id, p.name]))
  return (
    <figure className="space-y-1">
      <div className="relative overflow-hidden rounded-md bg-card shadow-ring">
        <img src={imageUrl} alt="Input photo" className="img-outline block w-full rounded-md" />
        {ground && (
          <svg viewBox={`0 0 ${ground.image_size[0]} ${ground.image_size[1]}`} className="absolute inset-0 h-full w-full">
            {ground.object && <MaskPath m={ground.object} className="fill-accent/10 stroke-accent" width={3} dashed />}
            {Object.entries(ground.parts).map(([id, m]) => (
              <g key={id} role="button" tabIndex={0} aria-label={`Select ${names.get(id) ?? id}`} className="cursor-pointer"
                onClick={() => onSelect(id)} onKeyDown={e => { if (e.key === 'Enter') onSelect(id) }}>
                <MaskPath m={m} width={2} className={id === selectedId ? 'fill-accent/40 stroke-accent' : 'fill-teal/20 stroke-teal hover:fill-teal/35'} />
              </g>
            ))}
          </svg>
        )}
      </div>
      {ground && (
        <figcaption className="font-mono text-caption text-muted">
          SAM 3.1 · object {ground.object ? 'found' : 'not found'} · {Object.keys(ground.parts).length} visible parts grounded
        </figcaption>
      )}
    </figure>
  )
}

function MaskPath({ m, className, width, dashed }: { m: Ground['parts'][string]; className: string; width: number; dashed?: boolean }) {
  const [l, t, r, b] = m.mask_bounds
  const [mw, mh] = m.mask_size
  return (
    <path d={m.mask_svg} transform={`translate(${l} ${t}) scale(${(r - l) / mw} ${(b - t) / mh})`}
      className={className} strokeWidth={width} vectorEffect="non-scaling-stroke" strokeDasharray={dashed ? '6 4' : undefined} />
  )
}

export function ObjectCard({ anatomy }: { anatomy: Anatomy }) {
  const o = anatomy.object
  const pct = Math.round(o.confidence * 100)
  return (
    <section aria-labelledby="object-name" className="space-y-1.5">
      <div>
        <p className="font-mono text-caption uppercase tracking-wider text-muted">{o.category}</p>
        <h2 id="object-name" className="text-title2 text-ink">{o.name}</h2>
        {(o.maker || o.model) && <p className="text-subheadline text-muted">{[o.maker, o.model].filter(Boolean).join(' · ')}</p>}
      </div>
      <p className="text-callout text-ink">{o.summary}</p>
      <dl className="grid grid-cols-2 gap-1 font-mono text-caption">
        <div className="rounded-sm bg-ink/5 p-1"><dt className="text-muted">Identification</dt><dd className="tabular text-ink">{pct}%</dd></div>
        <div className="rounded-sm bg-ink/5 p-1"><dt className="text-muted">Size (mm)</dt><dd className="tabular text-ink">{o.dimensions_mm.map(Math.round).join(' × ')}</dd></div>
      </dl>
      {o.evidence.length > 0 && (
        <details className="text-footnote text-muted">
          <summary className="cursor-pointer text-ink">Why Magneye thinks so</summary>
          <ul className="mt-1 list-disc space-y-px ps-2">{o.evidence.map(e => <li key={e}>{e}</li>)}</ul>
        </details>
      )}
    </section>
  )
}

export function AskBar({ disabled, onAsk, onTranscribe }: { disabled: boolean; onAsk: (q: string) => void; onTranscribe: (wav: Blob) => Promise<string> }) {
  const [q, setQ] = useState('')
  const [rec, setRec] = useState<Recorder | null>(null)
  const [listening, setListening] = useState<'idle' | 'recording' | 'transcribing'>('idle')
  const [micError, setMicError] = useState<string | null>(null)

  const toggleMic = async () => {
    setMicError(null)
    if (rec) {
      setListening('transcribing')
      try {
        const text = await onTranscribe(await rec.stop())
        setQ(text)
        if (text) onAsk(text)
      } catch (e) { setMicError(e instanceof Error ? e.message : String(e)) }
      setRec(null); setListening('idle')
      return
    }
    try { setRec(await startRecording()); setListening('recording') } catch (e) { setMicError(e instanceof Error ? e.message : String(e)) }
  }

  return (
    <form className="space-y-1" onSubmit={e => { e.preventDefault(); if (q.trim()) onAsk(q.trim()) }}>
      <label htmlFor="ask" className="text-headline text-ink">Ask about it</label>
      <div className="flex gap-1">
        <input id="ask" value={q} onChange={e => setQ(e.target.value)} disabled={disabled}
          placeholder={listening === 'recording' ? 'Listening…' : 'e.g. why does the stick drift?'}
          className="min-h-tap min-w-0 flex-1 rounded-sm bg-card px-1.5 text-body text-ink shadow-ring placeholder:text-muted disabled:opacity-50" />
        <button type="button" onClick={toggleMic} disabled={disabled || listening === 'transcribing'}
          aria-label={rec ? 'Stop recording' : 'Ask by voice'} title={rec ? 'Stop recording' : 'Ask by voice (Muse Voice Transcribe)'} aria-pressed={!!rec}
          className={`press grid h-6 w-6 shrink-0 place-items-center rounded-sm shadow-ring disabled:opacity-50 ${rec ? 'bg-red text-white' : 'bg-card text-ink hover:shadow-ring-hover'}`}>
          {listening === 'recording' ? <IconStop className="h-3 w-3" /> : listening === 'transcribing' ? <ThinkingOrb state="listening" size={20} aria-hidden="true" /> : <IconMic className="h-3 w-3" />}
        </button>
        <button type="submit" disabled={disabled || !q.trim()} aria-label="Ask" title="Ask"
          className="press grid h-6 w-6 shrink-0 place-items-center rounded-sm bg-accent-fill text-white hover:bg-accent-strong disabled:opacity-40">
          <IconArrow className="h-3 w-3" strokeWidth={2} />
        </button>
      </div>
      {micError && <p role="alert" className="text-footnote text-red-text">⚠ {micError}</p>}
    </form>
  )
}

export function AnswerCard({ answer, prompt }: { answer: Answer; prompt: string }) {
  return (
    <section aria-label="Answer" className="glass edge space-y-1 rounded-lg p-2">
      <p className="font-mono text-caption text-muted">“{prompt}”</p>
      <p className="text-body text-ink">{answer.text}</p>
    </section>
  )
}

export function ScenarioList({ scenarios, activeId, onPlay }: { scenarios: Scenario[]; activeId: string | null; onPlay: (s: Scenario) => void }) {
  if (!scenarios.length) return null
  return (
    <section aria-labelledby="scn" className="space-y-1">
      <h3 id="scn" className="text-headline text-ink">How it works</h3>
      <ul className="space-y-1">
        {scenarios.map(s => (
          <li key={s.id}>
            <button type="button" onClick={() => onPlay(s)} aria-pressed={s.id === activeId}
              className={`press flex min-h-tap w-full items-start gap-1.5 rounded-md p-1.5 shadow-ring hover:shadow-ring-hover ${s.id === activeId ? 'bg-accent/10' : 'bg-card'}`}>
              <IconPlay className={`mt-px h-2.5 w-2.5 shrink-0 ${s.id === activeId ? 'text-accent-text' : 'text-muted'}`} />
              <span className="min-w-0">
                <span className="block text-subheadline font-medium text-ink">{s.title}</span>
                <span className="block text-footnote text-muted">{s.summary}</span>
              </span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  )
}

const LIKELIHOOD = { common: 'bg-red/10 text-red-text', occasional: 'bg-orange/10 text-orange-text', rare: 'bg-ink/5 text-muted' } as const

export function FailureList({ failures, parts, activeId, onShow }: { failures: Failure[]; parts: Part[]; activeId: string | null; onShow: (f: Failure) => void }) {
  if (!failures.length) return null
  const name = (id: string) => parts.find(p => p.id === id)?.name ?? id
  return (
    <section aria-labelledby="fail" className="space-y-1">
      <h3 id="fail" className="text-headline text-ink">When it goes wrong</h3>
      <ul className="space-y-1">
        {failures.map(f => (
          <li key={f.id}>
            <button type="button" onClick={() => onShow(f)} aria-pressed={f.id === activeId}
              className={`press w-full space-y-px rounded-md p-1.5 shadow-ring hover:shadow-ring-hover ${f.id === activeId ? 'bg-red/10' : 'bg-card'}`}>
              <span className="flex items-start justify-between gap-1">
                <span className="text-subheadline font-medium text-ink">{f.symptom}</span>
                <span className={`shrink-0 rounded-full px-1 font-mono text-caption2 leading-5 ${LIKELIHOOD[f.likelihood]}`}>{f.likelihood}</span>
              </span>
              <span className="block text-footnote text-muted">Culprit: {name(f.culprit_part_id)}</span>
            </button>
            {f.id === activeId && (
              <div className="mt-1 space-y-1 rounded-md bg-card p-1.5 text-footnote text-ink shadow-ring">
                <p>{f.mechanism}</p>
                {f.checks.length > 0 && (<><p className="font-medium">Try first</p><ul className="list-disc space-y-px ps-2">{f.checks.map(c => <li key={c}>{c}</li>)}</ul></>)}
                <p><span className="font-medium">Fix:</span> {f.fix}</p>
                {f.access_path.length > 0 && (
                  <p className="font-mono text-caption text-muted">Remove: {f.access_path.map(name).join(' → ')}</p>
                )}
              </div>
            )}
          </li>
        ))}
      </ul>
    </section>
  )
}

export function PartDetail({ part, onClose }: { part: Part; onClose: () => void }) {
  const look = MATERIAL_LOOK[part.material] ?? MATERIAL_LOOK.other
  return (
    <section aria-labelledby="part-name" className="space-y-1 rounded-md bg-card p-1.5 shadow-ring">
      <div className="flex items-start justify-between gap-1">
        <h3 id="part-name" className="text-headline text-ink">{part.name}</h3>
        <button type="button" onClick={onClose} className="press text-footnote text-accent-text hover:underline">Close</button>
      </div>
      <p className="text-callout text-ink">{part.function}</p>
      <div className="flex flex-wrap gap-1 font-mono text-caption2 leading-5">
        <span className="inline-flex items-center gap-px rounded-full bg-ink/5 px-1 text-ink">
          <span className="me-px inline-block h-1 w-1 rounded-full" style={{ background: look.color }} aria-hidden="true" />{look.label}
        </span>
        <span className={`rounded-full px-1 ${part.grounded ? 'bg-teal/10 text-teal-text' : 'bg-ink/5 text-muted'}`}>
          {part.grounded ? 'Seen in photo · SAM 3.1' : part.kind === 'internal' ? 'Hidden · inferred' : 'Not visible in photo'}
        </span>
        <span className="tabular rounded-full bg-ink/5 px-1 text-ink">{Math.round(part.confidence * 100)}% sure</span>
      </div>
    </section>
  )
}
