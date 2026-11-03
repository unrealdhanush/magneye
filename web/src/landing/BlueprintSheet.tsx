import { useEffect, useRef, useState, type CSSProperties } from 'react'

const PARTS = [
  { id: 'shell', n: 1, name: 'Shell', how: 'seen in photo', size: '60 × 112 × 38' },
  { id: 'wheel', n: 2, name: 'Scroll wheel', how: 'seen in photo', size: '9 × 18' },
  { id: 'switch', n: 3, name: 'Click switches', how: 'documented', size: '15 × 10 each' },
  { id: 'board', n: 4, name: 'Circuit board', how: 'documented', size: '48 × 63' },
  { id: 'battery', n: 5, name: 'Battery', how: 'inferred', size: '29 × 27' },
] as const
type Id = (typeof PARTS)[number]['id']

const delay = (s: number) => ({ '--d': `${s}s` }) as CSSProperties

/** The hero's mouse as the app draws it in Blueprint: cyanotype, hidden parts dashed, numbered balloons. */
export function BlueprintSheet() {
  const [on, setOn] = useState<Id | null>(null)
  const [seen, setSeen] = useState(false)
  const box = useRef<HTMLDivElement>(null)

  // The sheet draws itself once, when it first scrolls into view.
  useEffect(() => {
    const el = box.current
    if (!el || !('IntersectionObserver' in window)) { setSeen(true); return }
    const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setSeen(true); io.disconnect() } }, { threshold: 0.35 })
    io.observe(el)
    return () => io.disconnect()
  }, [])

  const part = (id: Id, hidden: boolean) => ({
    className: `bp-part${hidden ? ' is-hidden' : ''}${on === id ? ' is-selected' : ''}`,
    onPointerEnter: () => setOn(id), onPointerLeave: () => setOn(null),
  })
  const balloon = (id: Id, n: number, x: number, y: number, dx: number, dy: number, d: number) => (
    <g className={`bp-balloon bp-fade${on === id ? ' is-selected' : ''}`} style={delay(d)}>
      <line className="bp-leader" x1={x} y1={y} x2={dx} y2={dy} />
      <circle className="bp-dot" cx={dx} cy={dy} r={2.5} />
      <circle className="bp-balloon-ring" cx={x} cy={y} r={11} />
      <text className="bp-balloon-text bp-num" x={x} y={y} fontSize={12}>{n}</text>
    </g>
  )

  return (
    <div ref={box} className={`lp-sheet${seen ? ' is-in' : ''}`}>
      <svg className="blueprint" viewBox="0 0 600 440" role="img" aria-label="Blueprint of a wireless mouse seen from above, with five numbered parts">
        <defs>
          <pattern id="lp-grid" width="20" height="20" patternUnits="userSpaceOnUse"><path d="M20 0H0V20" className="bp-gridline" /></pattern>
        </defs>
        <rect className="bp-paper" width="600" height="440" />
        <rect fill="url(#lp-grid)" width="600" height="440" />
        <rect className="bp-frame" x="12" y="12" width="576" height="416" />

        <g {...part('shell', false)}>
          <path className="bp-line bp-draw" pathLength={1} style={delay(0)}
            d="M160 150C160 90 200 58 250 58C300 58 340 90 340 150L346 330C348 372 312 394 250 394C188 394 152 372 154 330Z" />
          <path className="bp-line bp-fade" style={delay(0.9)} d="M250 58V170M163 190Q250 204 337 190" />
        </g>
        <g {...part('wheel', false)}>
          <rect className="bp-line bp-draw" pathLength={1} style={delay(0.5)} x="236" y="92" width="28" height="54" rx="12" />
        </g>
        <g {...part('board', true)}>
          <rect className="bp-line bp-fade" style={delay(1.1)} x="178" y="150" width="144" height="190" rx="14" />
        </g>
        <g {...part('battery', true)}>
          <rect className="bp-line bp-fade" style={delay(1.25)} x="206" y="232" width="88" height="82" rx="10" />
        </g>
        <g {...part('switch', true)}>
          <rect className="bp-line bp-fade" style={delay(1.4)} x="182" y="108" width="44" height="30" rx="5" />
          <rect className="bp-line bp-fade" style={delay(1.4)} x="274" y="108" width="44" height="30" rx="5" />
        </g>
        <path className="bp-center bp-fade" style={delay(1.5)} d="M250 40V412M130 226H370" />

        {/* Dimensions, in millimetres at 3 px each. */}
        <g className="bp-fade" style={delay(1.6)}>
          <line className="bp-ext" x1="160" y1="150" x2="160" y2="34" /><line className="bp-ext" x1="340" y1="150" x2="340" y2="34" />
          <line className="bp-dim" x1="160" y1="44" x2="340" y2="44" />
          <polygon className="bp-arrowhead" points="160,44 170,41 170,47" /><polygon className="bp-arrowhead" points="340,44 330,41 330,47" />
          <text className="bp-dimtext" x="250" y="36">60</text>
          <line className="bp-ext" x1="240" y1="58" x2="108" y2="58" /><line className="bp-ext" x1="240" y1="394" x2="108" y2="394" />
          <line className="bp-dim" x1="118" y1="58" x2="118" y2="394" />
          <polygon className="bp-arrowhead" points="118,58 115,68 121,68" /><polygon className="bp-arrowhead" points="118,394 115,384 121,384" />
          <text className="bp-dimtext" x="108" y="226" transform="rotate(-90 108 226)">112</text>
        </g>

        {balloon('shell', 1, 392, 70, 314, 82, 1.7)}
        {balloon('wheel', 2, 392, 110, 264, 100, 1.8)}
        {balloon('switch', 3, 392, 150, 318, 123, 1.9)}
        {balloon('board', 4, 392, 200, 322, 200, 2.0)}
        {balloon('battery', 5, 392, 270, 294, 270, 2.1)}

        <g className="bp-fade" style={delay(2.2)}>
          <rect className="bp-titleblock" x="372" y="336" width="208" height="80" />
          <text className="bp-title" x="386" y="366">Wireless mouse</text>
          <text className="bp-th" x="386" y="388">Top view, illustrative</text>
          <text className="bp-th" x="386" y="406">Dimensions in mm</text>
        </g>
      </svg>

      <div>
        <ol className="lp-plist" aria-label="Parts">
          {PARTS.map(p => (
            <li key={p.id}>
              <button type="button" className={`lp-prow${on === p.id ? ' is-on' : ''}`}
                onPointerEnter={() => setOn(p.id)} onPointerLeave={() => setOn(null)} onFocus={() => setOn(p.id)} onBlur={() => setOn(null)}>
                <span className="lp-bal">{p.n}</span>
                <span><span className="lp-prow-name">{p.name}</span><span className="lp-prow-how">{p.how}</span></span>
                <span className="lp-prow-size">{p.size}</span>
              </button>
            </li>
          ))}
        </ol>
        <p className="lp-fine">Sizes are illustrative. On your own object they come from the photo, cited sources and what Magneye infers.</p>
      </div>
    </div>
  )
}
