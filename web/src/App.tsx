import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api'
import { ThemeToggle } from './theme'
import type { Agent, AgentEvent, Anatomy, Failure, Ground, Playback, Scenario, Session, ViewerAction } from './types'
import { StartScreen } from './ui/StartScreen'
import { IconLayers, IconReset, IconXray } from './ui/icons'
import {
  AgentTrace, AskBar, Conversation, FailureList, ObjectCard, PartDetail, PhotoMasks, ScenarioList,
  type TraceStep, type Turn,
} from './ui/panels'
import { Viewer } from './viewer/Viewer'

const sessionFromHash = () => location.hash.match(/^#\/s\/([a-z0-9]+)$/)?.[1] ?? null
const message = (e: unknown) => (e instanceof Error ? e.message : String(e))

export default function App() {
  const [sid, setSid] = useState<string | null>(sessionFromHash)
  const [health, setHealth] = useState<{ has_key: boolean; sam3d: boolean } | null>(null)
  const [uploading, setUploading] = useState(false)
  const [startError, setStartError] = useState<string | null>(null)

  useEffect(() => { api.health().then(setHealth).catch(() => setHealth(null)) }, [])
  useEffect(() => {
    const onHash = () => setSid(sessionFromHash())
    addEventListener('hashchange', onHash)
    return () => removeEventListener('hashchange', onHash)
  }, [])

  const start = async (photo: Blob, prompt: string, extras: Blob[]) => {
    setUploading(true); setStartError(null)
    try {
      const s = await api.createSession(photo, prompt, extras)
      location.hash = `#/s/${s.id}`
    } catch (e) { setStartError(message(e)) }
    setUploading(false)
  }

  return (
    <>
      <a href="#main" className="sr-only focus:not-sr-only">Skip to content</a>
      <header className="glass sticky top-0 z-10 border-x-0 border-t-0">
        <div className={`mx-auto flex items-center justify-between gap-2 px-2 py-1 sm:px-3 ${sid ? '' : 'max-w-page'}`}>
          <h1 className="font-display text-headline tracking-wide">
            <a href="#/" className="text-ink no-underline hover:underline">MAGNEYE</a>
          </h1>
          <ThemeToggle />
        </div>
      </header>
      {/* The workspace uses the whole window; the start screen stays a readable width. */}
      <main id="main" className={`mx-auto px-2 pb-6 sm:px-3 ${sid ? '' : 'max-w-page'}`}>
        {health && !health.has_key && (
          <p role="alert" className="mt-2 rounded-md bg-card p-1.5 text-callout text-orange-text shadow-ring">
            ⚠ The server has no MODEL_API_KEY yet. Add your Meta Model API key to <code className="font-mono">.env</code> and restart it.
          </p>
        )}
        {sid ? <Workspace key={sid} sid={sid} sam3d={!!health?.sam3d} />
          : <StartScreen busy={uploading} error={startError} onStart={start} />}
      </main>
    </>
  )
}

function Workspace({ sid, sam3d }: { sid: string; sam3d: boolean }) {
  const [session, setSession] = useState<Session | null>(null)
  const [anatomy, setAnatomy] = useState<Anatomy | null>(null)
  const [ground, setGround] = useState<Ground | null>(null)
  const [meshUrl, setMeshUrl] = useState<string | null>(null)
  const [steps, setSteps] = useState<TraceStep[]>([])
  const [turns, setTurns] = useState<Turn[]>([])
  const [thinking, setThinking] = useState<Agent | null>(null)
  const [playback, setPlayback] = useState<Playback>({ kind: 'idle' })
  const [explode, setExplode] = useState(0)
  const [xray, setXray] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [focusIds, setFocusIds] = useState<string[]>([])
  const [assembleAt, setAssembleAt] = useState<number | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const started = useRef(false)
  // Latest anatomy/ground as streamed, readable after an await without waiting for a re-render.
  const latest = useRef<{ anatomy: Anatomy | null; ground: Ground | null }>({ anatomy: null, ground: null })

  const play = (s: Scenario) => { setPlayback({ kind: 'scenario', scenario: s, startedAt: performance.now() }); setFocusIds([]) }
  const show = (f: Failure) => { setPlayback({ kind: 'failure', failure: f, startedAt: performance.now() }); setFocusIds([]); setSelectedId(null) }

  const act = useCallback((a: ViewerAction) => {
    if (a.type === 'highlight') { setPlayback({ kind: 'idle' }); setFocusIds(a.part_ids); setExplode(e => Math.max(e, 0.45)) }
    else if (a.type === 'play_scenario') play(a.scenario)
    else if (a.type === 'show_failure') show(a.failure)
    else if (a.type === 'explode') setExplode(a.amount)
  }, [])

  /** Fold one agent event into the trace, the conversation and the viewer. `live` is false when replaying a saved trace. */
  const onEvent = useCallback((e: AgentEvent, live = true) => {
    switch (e.type) {
      case 'step':
        setThinking(null)
        setSteps(prev => {
          const i = prev.findIndex(st => st.status === 'running' && st.agent === e.agent && st.tool === e.tool)
          const next = { agent: e.agent, tool: e.tool, status: e.status, text: e.text }
          if (i >= 0 && e.status !== 'running') return prev.map((st, j) => (j === i ? next : st))
          return [...prev, next]
        })
        if (e.status !== 'running' && live) setThinking(e.tool === 'ground' || e.tool === 'finish' ? null : e.agent)
        break
      case 'thought': setSteps(prev => [...prev, { agent: e.agent, tool: 'thought', status: 'note', text: e.text }]); break
      case 'user': setTurns(t => [...t, { role: 'user', text: e.text }]); break
      case 'question': setThinking(null); setTurns(t => [...t, { role: 'question', text: e.text, options: e.options }]); break
      case 'answer': setThinking(null); setTurns(t => [...t, { role: 'answer', text: e.text }]); break
      case 'anatomy': latest.current.anatomy = e.anatomy; setAnatomy(e.anatomy); break
      case 'ground': latest.current.ground = e.ground; setGround(e.ground); break
      case 'action': if (live) act(e.action); break
      case 'error': setThinking(null); setSteps(prev => [...prev, { agent: 'guide', tool: 'error', status: 'error', text: e.message }]); break
      case 'done': setThinking(null); break
    }
  }, [act])

  const [asking, setAsking] = useState(false)
  const ask = useCallback(async (prompt: string) => {
    setAsking(true); setThinking('guide')
    try { await api.ask(sid, prompt, e => onEvent(e)) }
    catch (err) { onEvent({ type: 'error', message: message(err) }) }
    setAsking(false); setThinking(null)
  }, [sid, onEvent])

  // Load the session, replay its saved trace, then run whatever the agents have not done yet.
  useEffect(() => {
    if (started.current) return
    started.current = true
    ;(async () => {
      let s: Session
      try { s = await api.session(sid) } catch (e) { setLoadError(message(e)); return }
      setSession(s)
      s.trace.forEach(e => onEvent(e, false))
      if (!s.anatomy || !s.ground) {
        setThinking('builder')
        try { await api.build(sid, e => onEvent(e)) } catch (e) { onEvent({ type: 'error', message: message(e) }) }
        setThinking(null)
      } else {
        onEvent({ type: 'anatomy', anatomy: s.anatomy }, false)
        onEvent({ type: 'ground', ground: s.ground }, false)
      }
      if (!latest.current.anatomy) return
      // The photo pops out as a 3D shell, then opens up.
      setTimeout(() => setExplode(e => Math.max(e, 0.55)), 900)

      if (s.has_mesh) setMeshUrl(`/api/sessions/${sid}/shell.glb`)
      else if (sam3d && latest.current.ground?.object) {
        onEvent({ type: 'step', agent: 'builder', tool: 'shell', status: 'running' })
        api.reconstruct(sid)
          .then(r => { setMeshUrl(r.mesh_url); onEvent({ type: 'step', agent: 'builder', tool: 'shell', status: 'done', text: 'SAM 3D mesh replaced the extruded silhouette' }) })
          .catch(e => onEvent({ type: 'step', agent: 'builder', tool: 'shell', status: 'error', text: message(e) }))
      }
      if (s.prompt.trim() && !s.trace.some(e => e.type === 'user')) await ask(s.prompt)
    })()
  }, [sid, sam3d, ask, onEvent])

  if (loadError) return <p role="alert" className="mt-3 text-body text-red-text">⚠ {loadError}</p>
  if (!session) return null

  const selected = anatomy?.parts.find(p => p.id === selectedId) ?? null
  const activeScenario = playback.kind === 'scenario' ? playback.scenario.id : null
  const activeFailure = playback.kind === 'failure' ? playback.failure.id : null

  return (
    // Phones stack everything; tablets put the stage on top with the two panels side by side below; from xl the
    // panels flank the stage and grow with the window, and 3xl splits the right panel into two columns.
    <div className="grid gap-2 pt-2 md:grid-cols-2 xl:grid-cols-[clamp(300px,20vw,440px)_minmax(0,1fr)_clamp(340px,24vw,520px)] 3xl:grid-cols-[clamp(360px,17vw,480px)_minmax(0,1fr)_clamp(720px,32vw,960px)]">
      <aside className="order-2 min-w-0 space-y-2 xl:order-1 xl:h-[calc(100vh-112px)] xl:overflow-y-auto xl:pe-1" aria-label="Input and agent trace">
        <PhotoMasks imageUrl={session.image_url} ground={ground} anatomy={anatomy} selectedId={selectedId} onSelect={setSelectedId} />
        <AgentTrace steps={steps} thinking={thinking} />
        {anatomy && anatomy.caveats.length > 0 && (
          <details className="rounded-md bg-card p-1.5 text-footnote text-muted shadow-ring">
            <summary className="cursor-pointer text-ink">What Magneye isn’t sure about</summary>
            <ul className="mt-1 list-disc space-y-px ps-2">{anatomy.caveats.map(c => <li key={c}>{c}</li>)}</ul>
          </details>
        )}
      </aside>

      <section className="stage relative order-1 h-[60vh] min-w-0 min-h-[416px] overflow-hidden rounded-xl shadow-ring md:col-span-2 xl:order-2 xl:col-span-1 xl:h-[calc(100vh-112px)]" aria-label="3D exploded view">
        <div className="stage-grid absolute inset-0" aria-hidden="true" />
        <span className="hud-corner left-2 top-2 border-l-2 border-t-2" aria-hidden="true" />
        <span className="hud-corner right-2 top-2 border-r-2 border-t-2" aria-hidden="true" />
        <span className="hud-corner bottom-2 left-2 border-b-2 border-l-2" aria-hidden="true" />
        <span className="hud-corner bottom-2 right-2 border-b-2 border-r-2" aria-hidden="true" />
        {anatomy ? (
          <Viewer anatomy={anatomy} ground={ground} imageUrl={session.image_url} meshUrl={meshUrl} explode={explode} xray={xray}
            playback={playback} focusIds={focusIds} selectedId={selectedId} onSelect={setSelectedId} assembleAt={assembleAt} />
        ) : (
          <div className="absolute inset-0 grid place-items-center">
            <img src={session.image_url} alt="" className="max-h-[60%] max-w-[60%] rounded-md opacity-60" />
          </div>
        )}
        {anatomy && (
          <div className="glass absolute inset-x-2 bottom-2 flex flex-wrap items-center gap-1 rounded-lg p-1 sm:inset-x-auto sm:left-1/2 sm:w-max sm:-translate-x-1/2 sm:flex-nowrap">
            <label className="flex min-h-tap items-center gap-1 px-1 text-subheadline text-ink">
              <IconLayers className="h-2.5 w-2.5" /> Explode
              <input type="range" min={0} max={1} step={0.01} value={explode} onChange={e => setExplode(+e.target.value)}
                className="w-[128px] accent-[rgb(var(--accent))]" aria-valuetext={`${Math.round(explode * 100)}%`} />
            </label>
            <button type="button" onClick={() => setXray(x => !x)} aria-pressed={xray}
              className={`press inline-flex min-h-tap items-center gap-1 rounded-sm px-1.5 text-subheadline ${xray ? 'bg-accent-fill text-white' : 'text-ink hover:bg-ink/5'}`}>
              <IconXray className="h-2.5 w-2.5" /> X-ray
            </button>
            <button type="button" onClick={() => { setPlayback({ kind: 'idle' }); setFocusIds([]); setSelectedId(null); setExplode(0); setAssembleAt(performance.now()) }}
              className="press inline-flex min-h-tap items-center gap-1 rounded-sm px-1.5 text-subheadline text-ink hover:bg-ink/5">
              <IconReset className="h-2.5 w-2.5" /> Assemble
            </button>
          </div>
        )}
      </section>

      {/* Below 3xl the two groups dissolve (display: contents) and their items interleave by order: object, chat,
          then the rest. On 3xl they become side-by-side columns: details on the left, the conversation on the right. */}
      <aside className="order-3 flex min-w-0 flex-col gap-3 xl:h-[calc(100vh-112px)] xl:overflow-y-auto xl:pe-1 3xl:grid 3xl:grid-cols-2 3xl:items-start"
        aria-label="Object, conversation and scenarios">
        <div className="contents 3xl:block 3xl:space-y-3">
          {anatomy && <div className="order-1"><ObjectCard anatomy={anatomy} /></div>}
          {selected && <div className="order-4"><PartDetail part={selected} onClose={() => setSelectedId(null)} /></div>}
          {anatomy && <div className="order-5"><ScenarioList scenarios={anatomy.scenarios} activeId={activeScenario} onPlay={play} /></div>}
          {anatomy && <div className="order-6"><FailureList failures={anatomy.failures} parts={anatomy.parts} activeId={activeFailure} onShow={show} /></div>}
        </div>
        <div className="contents 3xl:sticky 3xl:top-0 3xl:block 3xl:space-y-3">
          <div className="order-2"><Conversation turns={turns} busy={asking} onReply={ask} /></div>
          <div className="order-3"><AskBar disabled={!anatomy || asking} onAsk={ask} onTranscribe={async wav => (await api.transcribe(wav)).text} /></div>
        </div>
      </aside>
    </div>
  )
}
