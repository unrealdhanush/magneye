import { useEffect, useRef, useState } from 'react'
import { CameraCapture } from './CameraCapture'
import { IconArrow, IconCamera, IconPhoto } from './icons'

const EXAMPLES = ['How does this work?', 'Why does it drift when I don’t touch it?', 'What happens when I press the trigger?', 'How do I get to the battery?']

export function StartScreen({ busy, error, onStart }: { busy: boolean; error: string | null; onStart: (photo: Blob, prompt: string) => void }) {
  const [photo, setPhoto] = useState<Blob | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [prompt, setPrompt] = useState('')
  const [camera, setCamera] = useState(false)
  const [dragging, setDragging] = useState(false)
  const file = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!photo) { setPreview(null); return }
    const url = URL.createObjectURL(photo)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [photo])

  const take = (b: Blob | undefined | null) => { if (b && b.type.startsWith('image/')) { setPhoto(b); setCamera(false) } }

  return (
    <section aria-labelledby="start-title" className="grid gap-4 py-6 lg:grid-cols-2 lg:items-center lg:gap-8">
      <div className="space-y-3">
        <p className="font-mono text-footnote uppercase tracking-widest text-accent-text">Photo in · anatomy out</p>
        <h2 id="start-title" className="max-w-[16ch] font-display text-largeTitle">See inside anything you can photograph.</h2>
        <p className="max-w-[52ch] text-body text-ink">
          Give Magneye one photo and a question. Muse Wonder works out what the object is and what’s inside it,
          SAM 3.1 finds it in your photo, and the parts fly apart in 3D. Ask how it works and watch the chain of
          parts play out. Describe a fault and the likely culprit lights up red.
        </p>
        <ol className="grid max-w-[60ch] gap-1 font-mono text-footnote text-ink sm:grid-cols-3">
          {[['01', 'Muse Wonder', 'identifies & infers internals'], ['02', 'SAM 3.1', 'grounds parts in the photo'], ['03', 'SAM 3D', 'reconstructs the shell']].map(([n, t, d]) => (
            <li key={n} className="glass rounded-md p-1.5">
              <span className="tabular text-accent-text">{n}</span> <span className="font-medium">{t}</span>
              <span className="block text-caption">{d}</span>
            </li>
          ))}
        </ol>
      </div>

      <form className="glass edge space-y-2 rounded-xl p-3" onSubmit={e => { e.preventDefault(); if (photo) onStart(photo, prompt) }}>
        {camera ? (
          <CameraCapture onCapture={take} onClose={() => setCamera(false)} />
        ) : (
          <div
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={e => { e.preventDefault(); setDragging(false); take(e.dataTransfer.files[0]) }}
            className={`relative overflow-hidden rounded-lg bg-card shadow-ring transition-[box-shadow] duration-150 ease-out ${dragging ? 'ring-2 ring-accent' : ''}`}>
            {preview ? (
              <img src={preview} alt="Your photo" className="img-outline aspect-video w-full rounded-lg object-contain" />
            ) : (
              <button type="button" onClick={() => file.current?.click()}
                className="flex aspect-video w-full flex-col items-start justify-end gap-1 p-3 text-ink hover:bg-ink/5">
                <IconPhoto className="h-6 w-6 text-accent-text" strokeWidth={1.25} />
                <span className="text-headline">Drop a photo, or choose one</span>
                <span className="text-subheadline text-muted">One object, filling most of the frame, works best.</span>
              </button>
            )}
          </div>
        )}
        <input ref={file} type="file" accept="image/*" className="sr-only" tabIndex={-1} onChange={e => take(e.target.files?.[0])} />

        {!camera && (
          <div className="flex flex-wrap gap-1">
            <button type="button" onClick={() => file.current?.click()}
              className="press inline-flex min-h-tap items-center gap-1 rounded-sm bg-card px-2 py-1.5 text-body text-ink shadow-ring hover:shadow-ring-hover">
              <IconPhoto className="h-3 w-3" /> {photo ? 'Change photo' : 'Choose photo'}
            </button>
            <button type="button" onClick={() => setCamera(true)}
              className="press inline-flex min-h-tap items-center gap-1 rounded-sm bg-card px-2 py-1.5 text-body text-ink shadow-ring hover:shadow-ring-hover">
              <IconCamera className="h-3 w-3" /> Use camera
            </button>
          </div>
        )}

        <label className="block space-y-1">
          <span className="text-headline">What do you want to see?</span>
          <textarea value={prompt} onChange={e => setPrompt(e.target.value)} rows={2} placeholder="How does this work?"
            className="block w-full resize-none rounded-md bg-card px-2 py-1.5 text-body text-ink shadow-ring placeholder:text-muted focus:shadow-ring-hover" />
        </label>
        <div className="flex flex-wrap gap-1" aria-label="Example prompts">
          {EXAMPLES.map(x => (
            <button key={x} type="button" onClick={() => setPrompt(x)}
              className="press rounded-full bg-accent/10 px-1.5 py-px text-footnote leading-6 text-accent-text hover:bg-accent/15">{x}</button>
          ))}
        </div>

        {error && <p role="alert" className="flex gap-1 rounded-md bg-red/10 p-1.5 text-callout text-red-text">⚠ {error}</p>}

        <button type="submit" disabled={!photo || busy}
          className="press inline-flex min-h-tap w-full items-center justify-between gap-1 rounded-sm bg-accent-fill py-1.5 pe-1.5 ps-2 text-headline text-white hover:bg-accent-strong disabled:opacity-40">
          <span>{busy ? 'Uploading…' : 'See inside'}</span> <IconArrow className="h-3 w-3" strokeWidth={2} />
        </button>
      </form>
    </section>
  )
}
