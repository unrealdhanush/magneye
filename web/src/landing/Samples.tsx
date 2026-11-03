import { useEffect, useState } from 'react'
import { IconArrow } from '../ui/icons'

/**
 * Photos to try Magneye on. Clicking one runs the real pipeline on it, exactly as if the visitor had uploaded it,
 * so it takes the same time and counts toward the same hourly limit. Images live in public/samples/; a sample whose
 * file is missing is simply not shown. `credit` is the photographer and licence, required by most open licences.
 */
export interface Sample { id: string; title: string; file: string; prompt: string; credit?: string }

export const SAMPLES: Sample[] = [
  { id: 'daytona', title: 'Rolex Daytona', file: '/samples/rolex-daytona.jpg', prompt: 'How does the chronograph work?' },
  { id: 'quest', title: 'Meta Quest', file: '/samples/meta-quest.jpg', prompt: 'How does it track my head and hands?' },
  { id: 'ferrari', title: 'Ferrari', file: '/samples/ferrari.jpg', prompt: 'What is under the skin and how does it make it go?' },
]

async function isImage(url: string): Promise<boolean> {
  try {
    const r = await fetch(url, { method: 'HEAD' })
    return r.ok && (r.headers.get('content-type') ?? '').startsWith('image/')
  } catch { return false }
}

export function Samples({ onStart, busy, error, heading = true }: {
  onStart: (photo: Blob, prompt: string, extras: Blob[]) => void
  busy: boolean; error: string | null; heading?: boolean
}) {
  const [shown, setShown] = useState<Sample[]>([])
  const [picked, setPicked] = useState<string | null>(null)
  const [failed, setFailed] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    Promise.all(SAMPLES.map(async s => ((await isImage(s.file)) ? s : null)))
      .then(x => { if (live) setShown(x.filter((s): s is Sample => !!s)) })
    return () => { live = false }
  }, [])

  if (!shown.length) return null

  const run = async (s: Sample) => {
    setPicked(s.id); setFailed(null)
    try {
      const r = await fetch(s.file)
      if (!r.ok) throw new Error('Could not load the photo.')
      onStart(await r.blob(), s.prompt, [])
    } catch (e) {
      setFailed(e instanceof Error ? e.message : 'Could not load the photo.')
      setPicked(null)
    }
  }

  return (
    <section className={heading ? 'lp-sec' : undefined} aria-labelledby={heading ? 'sm-title' : undefined} aria-label={heading ? undefined : 'Sample photos'}>
      {heading && (
        <>
          <h2 id="sm-title" className="lp-display lp-h2">No photo? Use one of ours.</h2>
          <p className="lp-sub">Pick a photo and Magneye takes it apart live, the same way it does yours. It takes a minute or two.</p>
        </>
      )}
      <ul className="sm-grid">
        {shown.map(s => (
          <li key={s.id}>
            <button type="button" className="sm-card glass edge press" onClick={() => run(s)} disabled={busy || picked !== null} aria-busy={picked === s.id}>
              <img src={s.file} alt="" loading="lazy" className="sm-img" />
              <span className="sm-body">
                <span className="sm-title">{s.title}</span>
                <span className="sm-q">“{s.prompt}”</span>
                <span className="ex-go">{picked === s.id ? 'Starting…' : 'Take it apart'} <IconArrow className="h-3 w-3" strokeWidth={2} /></span>
              </span>
            </button>
            {s.credit && <p className="sm-credit">{s.credit}</p>}
          </li>
        ))}
      </ul>
      {(failed ?? error) && <p role="alert" className="ex-err">{failed ?? error}</p>}
    </section>
  )
}
