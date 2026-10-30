import { useEffect, useRef, useState } from 'react'
import { ThinkingOrb, type OrbState } from 'thinking-orbs'
import { startRecording, type Recorder } from '../audio'
import type { Agent, Anatomy, Failure, Ground, Part, Scenario } from '../types'
import { MATERIAL_LOOK } from '../viewer/palette'
import { IconArrow, IconCheck, IconChevron, IconMic, IconPlay, IconStop, IconWarn } from './icons'

/** One row of the agent trace: a tool call with its live status, or a note the agent wrote between turns. */
export interface TraceStep { agent: Agent; tool: string; status: 'running' | 'done' | 'error' | 'note'; text?: string }

const TOOL_LABEL: Record<string, { label: string; model: string; orb: OrbState }> = {
  research: { label: 'Research teardowns', model: 'Wonder · web', orb: 'searching' },
  reference_views: { label: 'Find other angles', model: 'Wonder · Commons', orb: 'searching' },
  draft: { label: 'Draft anatomy', model: 'Muse Wonder', orb: 'solving' },
  cache: { label: 'Reuse anatomy', model: 'cache', orb: 'working' },
  find_in_photo: { label: 'Check the photo', model: 'SAM 3.1', orb: 'searching' },
  check_geometry: { label: 'Check the geometry', model: 'checks', orb: 'solving' },
  view_layout: { label: 'Review layout', model: 'Muse Wonder', orb: 'shaping' },
  update_parts: { label: 'Revise parts', model: 'Muse Wonder', orb: 'shaping' },
  add_part: { label: 'Add a missed part', model: 'Muse Wonder', orb: 'shaping' },
  remove_part: { label: 'Remove a part', model: 'Muse Wonder', orb: 'shaping' },
  finish: { label: 'Verified', model: 'Muse Wonder', orb: 'working' },
  ground: { label: 'Ground visible parts', model: 'SAM 3.1', orb: 'searching' },
  shell: { label: 'Reconstruct the shell', model: 'SAM 3D', orb: 'shaping' },
  highlight: { label: 'Point at parts', model: 'viewer', orb: 'working' },
  play_scenario: { label: 'Play a scenario', model: 'viewer', orb: 'working' },
  create_scenario: { label: 'Write a scenario', model: 'Muse Wonder', orb: 'composing' },
  show_failure: { label: 'Show the fault', model: 'viewer', orb: 'working' },
  create_failure: { label: 'Write a diagnosis', model: 'Muse Wonder', orb: 'composing' },
  set_explode: { label: 'Open it up', model: 'viewer', orb: 'working' },
  look_closer: { label: 'Look closer', model: 'SAM 3.1', orb: 'searching' },
  ask_user: { label: 'Ask you', model: 'Muse Wonder', orb: 'composing' },
  limit: { label: 'Turn limit', model: 'agent', orb: 'working' },
  verify: { label: 'Verify', model: 'Muse Wonder', orb: 'solving' },
  side_profile: { label: 'Trace the side profile', model: 'SAM 3.1', orb: 'shaping' },
  move_parts: { label: 'Move the body', model: 'Muse Wonder', orb: 'shaping' },
  place_in_photo: { label: 'Place parts as seen', model: 'Muse Wonder', orb: 'shaping' },
  model_parts: { label: 'Model parts in 3D', model: 'Wonder · Manifold', orb: 'shaping' },
  view_models: { label: 'Check models vs photo', model: 'Muse Wonder', orb: 'searching' },
  place_parts: { label: 'Fit parts to the photo', model: 'Muse Wonder', orb: 'shaping' },
  set_camera: { label: 'Correct the viewpoint', model: 'Muse Wonder', orb: 'solving' },
  clean_photo: { label: 'Clear what covers it', model: 'SAM 3.1 · Muse Image', orb: 'shaping' },
  thought: { label: 'Thinking', model: 'Muse Wonder', orb: 'solving' },
  error: { label: 'Error', model: '', orb: 'working' },
}

const AGENT_TONE: Record<Agent, string> = { builder: 'bg-indigo/10 text-indigo-text', guide: 'bg-teal/10 text-teal-text' }

export function AgentTrace({ steps, thinking }: { steps: TraceStep[]; thinking: Agent | null }) {
  const last = useRef<HTMLLIElement>(null)
  const [open, setOpen] = useState<Set<number>>(new Set())
  const [all, setAll] = useState(false)
  // Follow the newest step while an agent is working; a replayed trace stays scrolled to the top.
  const live = !!thinking || steps[steps.length - 1]?.status === 'running'
  useEffect(() => { if (live) last.current?.scrollIntoView({ block: 'nearest' }) }, [steps.length, live])
  const toggle = (i: number) => setOpen(prev => { const next = new Set(prev); if (next.has(i)) next.delete(i); else next.add(i); return next })
  return (
    <details open className="group/trace">
      <summary className="flex min-h-tap cursor-pointer list-none items-center gap-1 rounded-sm text-ink [&::-webkit-details-marker]:hidden">
        <IconChevron className="h-2 w-2 shrink-0 transition-transform duration-150 ease-out group-open/trace:rotate-90" />
        <h2 className="text-headline">Agent trace</h2>
        <span className="tabular font-mono text-caption text-muted">{steps.length}</span>
        <button type="button" onClick={e => { e.preventDefault(); setAll(a => !a); setOpen(new Set()) }}
          className="press ms-auto rounded-sm px-1 py-0.5 text-footnote text-accent-text hover:bg-accent/10 group-[:not([open])]/trace:hidden">
          {all ? 'Collapse all' : 'Expand all'}
        </button>
      </summary>
      <ol className="mt-1 space-y-px rounded-md bg-card p-1 shadow-ring" aria-live="polite">
        {steps.map((st, i) => {
          const meta = TOOL_LABEL[st.tool] ?? { label: st.tool, model: '', orb: 'working' as OrbState }
          const expanded = all || open.has(i)
          const tone = st.status === 'error' ? 'text-red-text' : st.status === 'note' ? 'text-ink' : 'text-muted'
          const head = (
            <>
              <span className="grid h-2.5 w-2.5 shrink-0 place-items-center pt-px">
                {st.status === 'running' ? <ThinkingOrb state={meta.orb} size={20} aria-hidden="true" />
                  : st.status === 'done' ? <IconCheck className="h-2 w-2 text-green-text" />
                  : st.status === 'note' ? <span className="h-1 w-1 rounded-full bg-ink/30" aria-hidden="true" />
                  : <IconWarn className="h-2 w-2 text-red-text" />}
              </span>
              <span className="min-w-0 flex-1">
                {/* The model tag drops to its own line when the panel is too narrow, rather than squeezing the label. */}
                <span className="flex flex-wrap items-baseline gap-x-1">
                  <span className={`shrink-0 rounded-full px-1 font-mono text-caption2 leading-4 ${AGENT_TONE[st.agent]}`}>{st.agent}</span>
                  <span className="text-footnote font-medium text-ink">{meta.label}</span>
                  <span className="ms-auto shrink-0 font-mono text-caption2 uppercase tracking-wider text-muted">{meta.model}</span>
                </span>
                {st.text && (
                  <span className={`block text-caption ${tone} ${expanded ? 'whitespace-pre-wrap break-words' : 'truncate'} ${st.status === 'note' ? 'italic' : ''}`}>
                    {st.text}
                  </span>
                )}
              </span>
            </>
          )
          return (
            <li key={i} ref={i === steps.length - 1 ? last : undefined}>
              {st.text ? (
                <button type="button" onClick={() => toggle(i)} aria-expanded={expanded}
                  className="flex w-full items-start gap-1 rounded-sm px-1 py-1 text-start hover:bg-ink/5">{head}</button>
              ) : (
                <div className="flex items-start gap-1 px-1 py-1">{head}</div>
              )}
            </li>
          )
        })}
        {thinking && (
          <li className="flex items-center gap-1 px-1 py-1" role="status">
            <ThinkingOrb state="solving" size={20} aria-hidden="true" />
            <span className="text-footnote text-muted">{thinking === 'builder' ? 'Builder is thinking…' : 'Guide is thinking…'}</span>
          </li>
        )}
        {!steps.length && !thinking && <li className="px-1 py-1 text-footnote text-muted">Waiting to start</li>}
      </ol>
    </details>
  )
}

/** The input photo with SAM 3.1's masks traced over it. */
export function PhotoMasks({ imageUrl, originalUrl, ground, anatomy, selectedId, onSelect }: {
  imageUrl: string; originalUrl?: string | null; ground: Ground | null; anatomy: Anatomy | null; selectedId: string | null
  onSelect: (id: string) => void
}) {
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
      {(ground || originalUrl) && (
        <figcaption className="space-y-px font-mono text-caption text-muted">
          {ground && <span className="block">SAM 3.1 · object {ground.object ? 'found' : 'not found'} · {Object.keys(ground.parts).length} visible parts grounded</span>}
          {originalUrl && (
            <span className="block">
              Muse Image removed the hand covering it; the hidden part is filled in, not seen ·{' '}
              <a href={originalUrl} target="_blank" rel="noopener noreferrer">original photo</a>
            </span>
          )}
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
      <Provenance anatomy={anatomy} />
      {o.evidence.length > 0 && (
        <details className="text-footnote text-muted">
          <summary className="cursor-pointer text-ink">Why Magneye thinks so</summary>
          <ul className="mt-1 list-disc space-y-px ps-2">{o.evidence.map(e => <li key={e}>{e}</li>)}</ul>
        </details>
      )}
    </section>
  )
}

const BASIS: Record<'seen' | 'documented' | 'inferred', { label: string; tone: string }> = {
  seen: { label: 'seen in photo', tone: 'bg-teal/10 text-teal-text' },
  documented: { label: 'documented', tone: 'bg-indigo/10 text-indigo-text' },
  inferred: { label: 'inferred', tone: 'bg-ink/5 text-muted' },
}

const hostOf = (url: string) => url.split('//')[1]?.split('/')[0].replace(/^www\./, '') ?? url

/** How much of the anatomy is seen, documented by cited sources, or inferred, plus the sources themselves. */
function Provenance({ anatomy }: { anatomy: Anatomy }) {
  const counts = { seen: 0, documented: 0, inferred: 0 }
  for (const p of anatomy.parts) counts[p.grounded ? 'seen' : p.basis === 'documented' ? 'documented' : p.basis === 'seen' ? 'seen' : 'inferred']++
  const sources = anatomy.research?.sources ?? []
  const refs = anatomy.references ?? []
  return (
    <div className="space-y-1">
      <div className="flex flex-wrap gap-1 font-mono text-caption2 leading-5">
        {(Object.keys(counts) as (keyof typeof counts)[]).filter(k => counts[k]).map(k => (
          <span key={k} className={`rounded-full px-1 ${BASIS[k].tone}`}><span className="tabular">{counts[k]}</span> {BASIS[k].label}</span>
        ))}
      </div>
      {refs.length > 0 && (
        <div className="space-y-0.5">
          <p className="text-footnote text-muted">Also built from {refs.length === 1 ? 'this view' : 'these views'}</p>
          <ul className="flex flex-wrap gap-1">
            {refs.map(r => {
              const body = <>
                {r.image_url
                  ? <img src={r.image_url} alt={`${r.view.replace('_', ' ')} view: ${r.title}`} loading="lazy"
                      className="h-8 w-10 rounded-sm bg-card object-cover shadow-ring" />
                  : <span className="grid h-8 w-10 place-items-center rounded-sm bg-card px-0.5 shadow-ring">{r.page_url ? hostOf(r.page_url) : 'Photo'}</span>}
                <span className="mt-px block truncate capitalize text-ink">{r.view.replace('_', ' ')}</span>
                <span className="block truncate">{r.origin === 'user' ? 'Your photo' : r.license ?? 'Web photo'}</span>
              </>
              const tip = [r.title, r.note, r.license && `${r.license}${r.author ? ` · ${r.author}` : ''}`].filter(Boolean).join('\n')
              return (
                <li key={r.page_url ?? r.view}>
                  {r.page_url
                    ? <a href={r.page_url} target="_blank" rel="noopener noreferrer" title={tip} className="block w-10 text-caption2 text-muted hover:text-ink">{body}</a>
                    : <div title={tip} className="w-10 text-caption2 text-muted">{body}</div>}
                </li>
              )
            })}
          </ul>
        </div>
      )}
      {sources.length > 0 && (
        <details className="text-footnote">
          <summary className="cursor-pointer text-ink">Blueprint sources ({sources.length})</summary>
          <ol className="mt-1 list-decimal space-y-px ps-2.5">
            {sources.map(src => (
              <li key={src.url}><a href={src.url} target="_blank" rel="noopener noreferrer" className="break-words">{src.title}</a></li>
            ))}
          </ol>
        </details>
      )}
    </div>
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

export type Turn = { role: 'user' | 'answer' | 'question'; text: string; options?: string[] }

export function Conversation({ turns, onReply, busy }: { turns: Turn[]; onReply: (text: string) => void; busy: boolean }) {
  if (!turns.length) return null
  const last = turns[turns.length - 1]
  return (
    <section aria-label="Conversation with the guide" className="space-y-1">
      {turns.map((t, i) => t.role === 'user' ? (
        <p key={i} className="ms-4 rounded-md bg-accent/10 px-1.5 py-1 text-callout text-ink">{t.text}</p>
      ) : (
        <div key={i} className={`glass edge space-y-1 rounded-lg p-1.5 ${t.role === 'question' ? 'me-2' : ''}`}>
          <p className="font-mono text-caption2 uppercase tracking-wider text-muted">{t.role === 'question' ? 'Guide asks' : 'Guide'}</p>
          <p className="text-body text-ink">{t.text}</p>
          {t.role === 'question' && t === last && !!t.options?.length && (
            <div className="flex flex-wrap gap-1">
              {t.options.map(o => (
                <button key={o} type="button" disabled={busy} onClick={() => onReply(o)}
                  className="press min-h-tap rounded-full bg-accent/10 px-1.5 text-footnote text-accent-text hover:bg-accent/15 disabled:opacity-40">{o}</button>
              ))}
            </div>
          )}
        </div>
      ))}
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
        <span className={`rounded-full px-1 ${part.grounded ? 'bg-teal/10 text-teal-text' : part.basis === 'documented' ? 'bg-indigo/10 text-indigo-text' : 'bg-ink/5 text-muted'}`}>
          {part.grounded ? 'Seen in photo · SAM 3.1' : part.basis === 'documented' ? 'Documented · cited sources' : part.kind === 'internal' ? 'Hidden · inferred' : 'Not visible in photo'}
        </span>
        <span className="tabular rounded-full bg-ink/5 px-1 text-ink">{Math.round(part.confidence * 100)}% sure</span>
      </div>
    </section>
  )
}
