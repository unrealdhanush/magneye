import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api'
import { ThemeToggle } from './theme'
import type { Anatomy, Answer, Failure, Ground, Playback, Scenario, Session } from './types'
import { StartScreen } from './ui/StartScreen'
import { IconLayers, IconReset, IconXray } from './ui/icons'
import {
  AnswerCard, AskBar, FailureList, ObjectCard, PartDetail, PhotoMasks, Pipeline, ScenarioList,
  type StageKey, type StageState,
} from './ui/panels'
import { Viewer } from './viewer/Viewer'

const FRESH: Record<StageKey, StageState> = {
  anatomy: { status: 'pending' }, ground: { status: 'pending' }, shell: { status: 'pending' }, answer: { status: 'pending' },
}
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

  const start = async (photo: Blob, prompt: string) => {
    setUploading(true); setStartError(null)
    try {
      const s = await api.createSession(photo, prompt)
      location.hash = `#/s/${s.id}`
    } catch (e) { setStartError(message(e)) }
    setUploading(false)
  }

  return (
    <>
      <a href="#main" className="sr-only focus:not-sr-only">Skip to content</a>
      <header className="glass sticky top-0 z-10 border-x-0 border-t-0">
        <div className="mx-auto flex max-w-page items-center justify-between gap-2 px-2 py-1 sm:px-3">
          <h1 className="font-display text-headline tracking-wide">
            <a href="#/" className="text-ink no-underline hover:underline">MAGNEYE</a>
          </h1>
          <ThemeToggle />
        </div>
      </header>
      <main id="main" className="mx-auto max-w-page px-2 pb-6 sm:px-3">
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
  const [stages, setStages] = useState(FRESH)
  const [answer, setAnswer] = useState<{ prompt: string; answer: Answer } | null>(null)
  const [playback, setPlayback] = useState<Playback>({ kind: 'idle' })
  const [explode, setExplode] = useState(0)
  const [xray, setXray] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [focusIds, setFocusIds] = useState<string[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)
  const started = useRef(false)

  const stage = (key: StageKey, value: StageState) => setStages(s => ({ ...s, [key]: value }))

  const play = (s: Scenario) => { setPlayback({ kind: 'scenario', scenario: s, startedAt: performance.now() }); setFocusIds([]) }
  const show = (f: Failure) => { setPlayback({ kind: 'failure', failure: f, startedAt: performance.now() }); setFocusIds([]); setSelectedId(null) }

  const applyAnswer = useCallback((a: Answer, current: Anatomy) => {
    const scenarios = a.new_scenario ? [...current.scenarios, a.new_scenario] : current.scenarios
    const failures = a.new_failure ? [...current.failures, a.new_failure] : current.failures
    if (a.new_scenario || a.new_failure) setAnatomy({ ...current, scenarios, failures })
    const sc = scenarios.find(s => s.id === a.scenario_id)
    const fl = failures.find(f => f.id === a.failure_id)
    if (fl) show(fl)
    else if (sc) play(sc)
    else { setPlayback({ kind: 'idle' }); setFocusIds(a.focus_part_ids); setExplode(e => Math.max(e, 0.45)) }
  }, [])

  const ask = useCallback(async (prompt: string, current: Anatomy) => {
    stage('answer', { status: 'running' })
    try {
      const a = await api.ask(sid, prompt)
      setAnswer({ prompt, answer: a })
      applyAnswer(a, current)
      stage('answer', { status: 'done', detail: a.intent === 'symptom' ? 'Diagnosed a likely culprit' : a.intent === 'scenario' ? 'Playing a scenario' : 'Answered' })
    } catch (e) { stage('answer', { status: 'error', detail: message(e) }) }
  }, [sid, applyAnswer])

  // Run whichever stages this session has not finished yet.
  useEffect(() => {
    if (started.current) return
    started.current = true
    ;(async () => {
      let s: Session
      try { s = await api.session(sid) } catch (e) { setLoadError(message(e)); return }
      setSession(s)
      let a = s.anatomy, g = s.ground
      if (a) stage('anatomy', { status: 'done', detail: a.object.name })
      else {
        stage('anatomy', { status: 'running' })
        try { a = await api.anatomy(sid); stage('anatomy', { status: 'done', detail: `${a.object.name} · ${a.parts.length} parts` }) }
        catch (e) { stage('anatomy', { status: 'error', detail: message(e) }); return }
      }
      setAnatomy(a)
      if (g) stage('ground', { status: 'done', detail: `${Object.keys(g.parts).length} parts grounded` })
      else {
        stage('ground', { status: 'running' })
        try {
          const r = await api.ground(sid)
          a = r.anatomy; g = r.ground
          setAnatomy(a)
          stage('ground', { status: g.object ? 'done' : 'error', detail: g.object ? `${Object.keys(g.parts).length} visible parts grounded` : 'Object not found in photo; showing inferred layout' })
        } catch (e) { stage('ground', { status: 'error', detail: message(e) }) }
      }
      setGround(g)
      // The photo pops out as a 3D shell, then opens up.
      setTimeout(() => setExplode(0.55), 900)

      if (s.has_mesh) { setMeshUrl(`/api/sessions/${sid}/shell.glb`); stage('shell', { status: 'done', detail: 'Mesh loaded' }) }
      else if (sam3d && g?.object) {
        stage('shell', { status: 'running' })
        api.reconstruct(sid)
          .then(r => { setMeshUrl(r.mesh_url); stage('shell', { status: 'done', detail: 'Mesh replaced the extruded silhouette' }) })
          .catch(e => stage('shell', { status: 'error', detail: message(e) }))
      } else stage('shell', { status: 'skipped', detail: g?.object ? 'Using the photo silhouette shell' : 'Needs an object mask' })

      if (s.prompt.trim()) await ask(s.prompt, a)
      else stage('answer', { status: 'skipped', detail: 'No prompt given' })
    })()
  }, [sid, sam3d, ask])

  if (loadError) return <p role="alert" className="mt-3 text-body text-red-text">⚠ {loadError}</p>
  if (!session) return null

  const selected = anatomy?.parts.find(p => p.id === selectedId) ?? null
  const activeScenario = playback.kind === 'scenario' ? playback.scenario.id : null
  const activeFailure = playback.kind === 'failure' ? playback.failure.id : null

  return (
    <div className="grid gap-2 pt-2 lg:grid-cols-[320px_minmax(0,1fr)_360px]">
      <aside className="order-2 space-y-2 lg:order-1" aria-label="Input and pipeline">
        <PhotoMasks imageUrl={session.image_url} ground={ground} anatomy={anatomy} selectedId={selectedId} onSelect={setSelectedId} />
        <Pipeline stages={stages} />
        {anatomy && anatomy.caveats.length > 0 && (
          <details className="rounded-md bg-card p-1.5 text-footnote text-muted shadow-ring">
            <summary className="cursor-pointer text-ink">What Magneye isn’t sure about</summary>
            <ul className="mt-1 list-disc space-y-px ps-2">{anatomy.caveats.map(c => <li key={c}>{c}</li>)}</ul>
          </details>
        )}
      </aside>

      <section className="stage relative order-1 h-[60vh] min-h-[416px] overflow-hidden rounded-xl shadow-ring lg:order-2 lg:h-[calc(100vh-112px)]" aria-label="3D exploded view">
        <div className="stage-grid absolute inset-0" aria-hidden="true" />
        <span className="hud-corner left-2 top-2 border-l-2 border-t-2" aria-hidden="true" />
        <span className="hud-corner right-2 top-2 border-r-2 border-t-2" aria-hidden="true" />
        <span className="hud-corner bottom-2 left-2 border-b-2 border-l-2" aria-hidden="true" />
        <span className="hud-corner bottom-2 right-2 border-b-2 border-r-2" aria-hidden="true" />
        {anatomy ? (
          <Viewer anatomy={anatomy} ground={ground} imageUrl={session.image_url} meshUrl={meshUrl} explode={explode} xray={xray}
            playback={playback} focusIds={focusIds} selectedId={selectedId} onSelect={setSelectedId} />
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
            <button type="button" onClick={() => { setPlayback({ kind: 'idle' }); setFocusIds([]); setSelectedId(null); setExplode(0) }}
              className="press inline-flex min-h-tap items-center gap-1 rounded-sm px-1.5 text-subheadline text-ink hover:bg-ink/5">
              <IconReset className="h-2.5 w-2.5" /> Assemble
            </button>
          </div>
        )}
      </section>

      <aside className="order-3 space-y-3" aria-label="Object, answers and scenarios">
        {anatomy && <ObjectCard anatomy={anatomy} />}
        <AskBar disabled={!anatomy || stages.answer.status === 'running'} onAsk={q => anatomy && ask(q, anatomy)}
          onTranscribe={async wav => (await api.transcribe(wav)).text} />
        {answer && <AnswerCard prompt={answer.prompt} answer={answer.answer} />}
        {selected && <PartDetail part={selected} onClose={() => setSelectedId(null)} />}
        {anatomy && <ScenarioList scenarios={anatomy.scenarios} activeId={activeScenario} onPlay={play} />}
        {anatomy && <FailureList failures={anatomy.failures} parts={anatomy.parts} activeId={activeFailure} onShow={show} />}
      </aside>
    </div>
  )
}
