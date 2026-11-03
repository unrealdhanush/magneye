import { useEffect, useState } from 'react'
import { api } from '../api'
import type { Example } from '../types'
import { IconArrow } from '../ui/icons'

const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`

/**
 * Finished examples that open in one click, so Magneye can be tried without a photo.
 * Each one opens as the visitor's own copy. Renders nothing when the server has no examples (local development).
 */
export function Examples({ heading = true }: { heading?: boolean }) {
  const [items, setItems] = useState<Example[]>([])
  const [opening, setOpening] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    api.examples().then(x => { if (live) setItems(x) }).catch(() => {})
    return () => { live = false }
  }, [])

  if (!items.length) return null

  const open = async (id: string) => {
    setOpening(id); setError(null)
    try {
      const { id: sid } = await api.openExample(id)
      location.hash = `#/s/${sid}`
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not open the example.')
      setOpening(null)
    }
  }

  return (
    <section className={heading ? 'lp-sec' : undefined} aria-labelledby={heading ? 'ex-title' : undefined} aria-label={heading ? undefined : 'Examples'}>
      {heading && (
        <>
          <h2 id="ex-title" className="lp-display lp-h2">Open one that’s already taken apart.</h2>
          <p className="lp-sub">No photo needed. Each example opens as your own copy, so you can ask it questions and edit it freely.</p>
        </>
      )}
      <ul className="ex-grid">
        {items.map(x => (
          <li key={x.id}>
            <button type="button" className="ex-card glass edge press" onClick={() => open(x.id)} disabled={opening !== null} aria-busy={opening === x.id}>
              <img src={x.image_url} alt="" loading="lazy" className="ex-img" />
              <span className="ex-body">
                <span className="ex-title">{x.title}</span>
                {x.blurb && <span className="ex-blurb">{x.blurb}</span>}
                <span className="ex-stats">
                  {plural(x.parts, 'part', 'parts')} · {plural(x.scenarios, 'walkthrough', 'walkthroughs')} · {plural(x.failures, 'fault', 'faults')}
                </span>
                <span className="ex-go">{opening === x.id ? 'Opening…' : 'Open this example'} <IconArrow className="h-3 w-3" strokeWidth={2} /></span>
              </span>
            </button>
          </li>
        ))}
      </ul>
      {error && <p role="alert" className="ex-err">{error}</p>}
    </section>
  )
}
