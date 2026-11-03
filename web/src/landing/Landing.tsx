import type { SVGProps } from 'react'
import { IconArrow } from '../ui/icons'
import { BlueprintSheet } from './BlueprintSheet'
import { Hero } from './Hero'
import './landing.css'

const STEPS = [
  ['Photograph it', 'Any object, from any side. Add up to five more photos from other angles or close up if you have them.'],
  ['Magneye finds the parts', 'It identifies the product, reads teardowns and specs, then locates each part it can see in your photo. The rest it works out and marks as inferred.'],
  ['Ask what you want to know', 'Ask how it works and watch the parts pass the signal along. Describe a fault and the likely culprit lights up red. Ask for a part to be added or moved.'],
]

const mark = (p: SVGProps<SVGSVGElement>) => ({ viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, 'aria-hidden': true, ...p })
const MARKS = [
  ['Seen in photo', 'SAM 3.1', 'SAM 3.1 located it in your picture, so its place comes from what the camera saw.', <svg key="s" {...mark({})}><circle cx="12" cy="12" r="7" fill="currentColor" /></svg>],
  ['Documented', 'cited sources', 'A teardown, spec sheet or manual that Magneye cited says it is there. Those sources are linked from the object card.', <svg key="d" {...mark({})}><circle cx="12" cy="12" r="8" /><path d="m8.5 12.2 2.4 2.4 4.6-5" /></svg>],
  ['Hidden', 'inferred', 'Nothing shows it, so Magneye reasons it out from how this kind of object is built. Treat it as an informed guess.', <svg key="h" {...mark({})}><circle cx="12" cy="12" r="8" strokeDasharray="3 3" /></svg>],
] as const

export function Landing() {
  return (
    <div className="lp">
      <Hero />

      <section className="lp-sec" aria-labelledby="lp-steps-title">
        <h2 id="lp-steps-title" className="lp-display lp-h2">One photo in. A model you can take apart out.</h2>
        <ol className="lp-steps">
          {STEPS.map(([t, d], i) => (
            <li key={t}><span className="lp-step-n">{i + 1}</span><h3>{t}</h3><p>{d}</p></li>
          ))}
        </ol>
      </section>

      <section className="lp-sec lp-know" aria-labelledby="lp-know-title">
        <div>
          <h2 id="lp-know-title" className="lp-display lp-h2">Every part says how it’s known.</h2>
          <p className="lp-sub">Magneye doesn’t pass off a guess as a measurement. Each part carries one of three marks, on the object card and in the blueprint.</p>
        </div>
        <ul className="lp-marks">
          {MARKS.map(([t, tag, d, icon]) => (
            <li key={t}>{icon}<div><h3>{t}<span>{tag}</span></h3><p>{d}</p></div></li>
          ))}
        </ul>
      </section>

      <section className="lp-sec" aria-labelledby="lp-bp-title">
        <h2 id="lp-bp-title" className="lp-display lp-h2">The same anatomy as an engineering sheet.</h2>
        <p className="lp-sub">Switch to Blueprint for front, top and right views, hidden parts dashed, dimensions and a parts list. Edit a part there and the 3D view follows.</p>
        <BlueprintSheet />
      </section>

      <section className="lp-sec" aria-labelledby="lp-cta-title">
        <div className="lp-cta glass edge">
          <h2 id="lp-cta-title" className="lp-display lp-h2" style={{ maxWidth: '24ch' }}>Photograph something you’ve never opened.</h2>
          <p className="lp-sub" style={{ margin: 0 }}>One object filling most of the frame works best.</p>
          <a className="lp-btn" href="#/new">Try it with a photo <IconArrow className="h-3 w-3" strokeWidth={2} /></a>
        </div>
      </section>

      <footer className="lp-foot">
        <p>Built for the Meta Global AI Developer Hackathon with Muse Wonder, SAM 3.1, Muse Image, SAM 3D Objects and Muse Voice Transcribe.</p>
      </footer>
    </div>
  )
}
