import { Component, lazy, Suspense, useEffect, useRef, useState, type CSSProperties, type ReactNode, type RefObject } from 'react'
import { IconArrow } from '../ui/icons'
import type { SceneState } from './MouseScene'

const MouseScene = lazy(() => import('./MouseScene'))

function webglOk() {
  try { return !!document.createElement('canvas').getContext('webgl2') } catch { return false }
}

/** If the 3D scene throws (lost context, old GPU), say so once and let the CSS illustration carry on. */
class Boundary extends Component<{ onError: () => void; children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  componentDidCatch() { this.props.onError() }
  render() { return this.state.failed ? null : this.props.children }
}

// Top to bottom, numbered the way the blueprint sheet numbers them. `lift` is the part's layer in the stack.
const PARTS = [
  { id: 'shell', n: 1, name: 'Shell', how: 'seen in photo', lift: 4 },
  { id: 'wheel', n: 2, name: 'Scroll wheel', how: 'seen in photo', lift: 3 },
  { id: 'switch', n: 3, name: 'Click switches', how: 'documented', lift: 2 },
  { id: 'board', n: 4, name: 'Circuit board', how: 'documented', lift: 1 },
  { id: 'battery', n: 5, name: 'Battery', how: 'inferred', lift: 0 },
] as const

const vars = (o: Record<string, string | number>) => o as CSSProperties
const smooth = (t: number) => { const c = Math.min(1, Math.max(0, t)); return c * c * (3 - 2 * c) }

/**
 * Drives the scene from scroll. Scroll sets a target; a short frame loop eases the CSS variables toward it,
 * so the parts glide instead of tracking the wheel in steps. Nothing re-renders React per frame.
 * Reduced motion gets the finished scene, still.
 */
function useScene(root: RefObject<HTMLElement | null>, shared: SceneState) {
  useEffect(() => {
    const el = root.current
    if (!el) return
    // The pinned scene sits just under the sticky header, whatever height it has.
    const head = document.querySelector('header')
    const measure = () => el.style.setProperty('--lp-head', `${Math.ceil(head?.getBoundingClientRect().height ?? 57)}px`)
    measure()
    const setPhase = (p: number) => { shared.phase = p; if (el.dataset.phase !== String(p)) el.dataset.phase = String(p) }
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
      el.classList.add('is-still'); el.style.setProperty('--e', '0.85'); shared.e = 0.85; setPhase(2)
      return
    }
    const cur = { e: 0, px: 0, py: 0 }, tgt = { e: 0, px: 0, py: 0 }
    let raf = 0
    const tick = () => {
      let moving = false
      for (const k of ['e', 'px', 'py'] as const) {
        const d = tgt[k] - cur[k]
        if (Math.abs(d) < 0.0005) cur[k] = tgt[k]
        else { cur[k] += d * 0.1; moving = true }
        el.style.setProperty(`--${k}`, cur[k].toFixed(4)); shared[k] = cur[k]
      }
      raf = moving ? requestAnimationFrame(tick) : 0
    }
    const kick = () => { if (!raf) raf = requestAnimationFrame(tick) }
    const onScroll = () => {
      measure()
      const r = el.getBoundingClientRect(), span = r.height - innerHeight
      const p = span > 0 ? Math.min(1, Math.max(0, -r.top / span)) : 0
      tgt.e = smooth((p - 0.04) / 0.5)
      setPhase(p < 0.1 ? 0 : p < 0.62 ? 1 : 2)
      kick()
    }
    const onMove = (ev: PointerEvent) => {
      tgt.px = (ev.clientX / innerWidth - 0.5) * 2
      tgt.py = (ev.clientY / innerHeight - 0.5) * 2
      kick()
    }
    onScroll()
    addEventListener('scroll', onScroll, { passive: true })
    addEventListener('resize', onScroll)
    addEventListener('pointermove', onMove, { passive: true })
    return () => {
      removeEventListener('scroll', onScroll); removeEventListener('resize', onScroll); removeEventListener('pointermove', onMove)
      cancelAnimationFrame(raf)
    }
  }, [root, shared])
}

function CssStage() {
  return (
    <div className="lp-scene" aria-hidden="true">
      <div className="lp-ground" />
      <div className="lp-layer" style={vars({ '--i': 0 })}><div className="lp-face lp-battery" /></div>
      <div className="lp-layer" style={vars({ '--i': 1 })}>
        <div className="lp-face lp-board"><i className="lp-chip" /><i className="lp-chip" /><i className="lp-chip" /></div>
      </div>
      <div className="lp-layer lp-layer--sw" style={vars({ '--i': 2 })}><i className="lp-sw lp-sw--l" /><i className="lp-sw lp-sw--r" /></div>
      <div className="lp-layer" style={vars({ '--i': 3 })}><i className="lp-wheel" /></div>
      <div className="lp-layer lp-layer--shell" style={vars({ '--i': 4 })}><div className="lp-face lp-shell" /></div>
    </div>
  )
}

export function Hero() {
  const root = useRef<HTMLElement>(null)
  const stage = useRef<HTMLDivElement>(null)
  const labels = useRef<HTMLUListElement>(null)
  const shared = useRef<SceneState>({ e: 0, px: 0, py: 0, phase: 0 }).current
  const [use3d, setUse3d] = useState(webglOk)
  const [ready, setReady] = useState(false)
  const [visible, setVisible] = useState(true)
  const [still] = useState(() => matchMedia('(prefers-reduced-motion: reduce)').matches)
  useScene(root, shared)
  // Stop drawing while the hero is off screen.
  useEffect(() => {
    const el = stage.current
    if (!el) return
    const io = new IntersectionObserver(([x]) => setVisible(!!x?.isIntersecting), { rootMargin: '120px' })
    io.observe(el)
    return () => io.disconnect()
  }, [])

  return (
    <section ref={root} data-phase="0" aria-labelledby="lp-title" className="lp-hero">
      <div className="lp-pin">
        <div className="lp-left">
          <div className="lp-copy">
            <h2 id="lp-title" className="lp-display lp-title">See inside anything you can photograph.</h2>
            <p className="lp-lede">
              Give Magneye one photo and a question. It works out what the object is, finds its parts in your picture,
              and pulls them apart in 3D.
            </p>
            <div className="lp-actions"><a className="lp-btn" href="#/new">Try it with a photo <IconArrow className="h-3 w-3" strokeWidth={2} /></a></div>
          </div>

          <div className="lp-cap">
            <p className="lp-cap-0">Scroll to take it apart. This is an illustration of a wireless mouse. Your photo becomes the real model.</p>
            <p className="lp-cap-1">Parts lift outermost first, so you can see what sits under what. Each one is numbered, named and marked with how Magneye knows it’s there.</p>
            <div className="lp-cap-2">
              <p className="lp-q">Why does one click register twice?</p>
              <p className="lp-a glass"><strong>The left switch.</strong> Worn contacts bounce, and a bounce sends a second click. The shell is lifted away and the switch is marked red.</p>
            </div>
          </div>
          <div className="lp-still-chat">
            <p className="lp-q">Why does one click register twice?</p>
            <p className="lp-a glass"><strong>The left switch.</strong> Worn contacts bounce, and a bounce sends a second click. The shell is lifted away and the switch is marked red.</p>
          </div>
        </div>

        <div ref={stage} className={`lp-stage${use3d && ready ? ' is-3d' : ''}`}>
          {!(use3d && ready) && <CssStage />}
          {use3d && (
            <Boundary onError={() => setUse3d(false)}>
              <Suspense fallback={null}>
                <MouseScene state={shared} labels={labels} visible={visible} still={still} onReady={() => setReady(true)} />
              </Suspense>
            </Boundary>
          )}
          <ul ref={labels} className="lp-labels" aria-hidden="true">
            {PARTS.map(p => (
              <li key={p.id} data-id={p.id} className="lp-label" style={vars({ '--i': p.lift })}>
                <span className="lp-bal">{p.n}</span>
                <span><span className="lp-lname">{p.name}</span><span className="lp-lhow">{p.how}</span></span>
              </li>
            ))}
          </ul>
          <ol className="sr-only">
            {PARTS.map(p => <li key={p.id}>{p.name}, {p.how}</li>)}
          </ol>
        </div>
      </div>
    </section>
  )
}
