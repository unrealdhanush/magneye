import { useEffect, useRef, useState } from 'react'
import { CameraCapture } from './CameraCapture'
import { IconArrow, IconCamera, IconPhoto, IconX } from './icons'

const MAX_EXTRAS = 5
const EXAMPLES = ['How does this work?', 'Why does it drift when I don’t touch it?', 'What happens when I press the trigger?', 'How do I get to the battery?']

/** An object URL for a blob, revoked when the blob changes or the component unmounts. */
function usePreview(blob: Blob | null) {
  const [url, setUrl] = useState<string | null>(null)
  useEffect(() => {
    if (!blob) { setUrl(null); return }
    const u = URL.createObjectURL(blob)
    setUrl(u)
    return () => URL.revokeObjectURL(u)
  }, [blob])
  return url
}

/** A removable thumbnail of one extra photo. */
function ExtraThumb({ photo, n, onRemove }: { photo: Blob; n: number; onRemove: () => void }) {
  const url = usePreview(photo)
  return (
    <li className="relative">
      {url && <img src={url} alt={`Extra photo ${n}`} className="img-outline h-8 w-10 rounded-sm object-cover" />}
      <button type="button" onClick={onRemove} aria-label={`Remove extra photo ${n}`} title="Remove"
        className="press absolute -end-1 -top-1 grid h-3 w-3 place-items-center rounded-full bg-card text-ink shadow-ring after:absolute after:-inset-1.5 after:content-[''] hover:shadow-ring-hover">
        <IconX className="h-1.5 w-1.5" strokeWidth={2} />
      </button>
    </li>
  )
}

export function StartScreen({ busy, error, onStart }: {
  busy: boolean; error: string | null; onStart: (photo: Blob, prompt: string, extras: Blob[]) => void
}) {
  const [photo, setPhoto] = useState<Blob | null>(null)
  const [extras, setExtras] = useState<Blob[]>([])
  const preview = usePreview(photo)
  const [prompt, setPrompt] = useState('')
  // Which photo the live camera is taking, if it is open: the main one or another extra.
  const [camera, setCamera] = useState<'main' | 'extra' | null>(null)
  const [dragging, setDragging] = useState(false)
  const file = useRef<HTMLInputElement>(null), extraFiles = useRef<HTMLInputElement>(null)

  const isImage = (b: Blob | undefined | null): b is Blob => !!b && b.type.startsWith('image/')
  const take = (b: Blob | undefined | null) => { if (isImage(b)) { setPhoto(b); setCamera(null) } }
  const addExtras = (bs: Iterable<Blob>) => {
    const fresh = [...bs].filter(isImage)
    if (fresh.length) setExtras(prev => [...prev, ...fresh].slice(0, MAX_EXTRAS))
    setCamera(null)
  }
  // Several files dropped at once: the first becomes the main photo (if there is none yet), the rest extras.
  const drop = (files: FileList) => {
    const images = [...files].filter(isImage)
    if (!photo && images.length) take(images.shift())
    addExtras(images)
  }

  return (
    <section aria-labelledby="start-title" className="grid gap-4 py-6 lg:grid-cols-2 lg:items-center lg:gap-8">
      <div className="space-y-3">
        <p className="font-mono text-footnote uppercase tracking-widest text-accent-text">Photo in · anatomy out</p>
        <h2 id="start-title" className="max-w-[16ch] font-display text-largeTitle">See inside anything you can photograph.</h2>
        <p className="max-w-[52ch] text-body text-ink">
          Give Magneye a photo (or a few from different sides) and a question. Muse Wonder works out what the object is and what’s inside it,
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

      <form className="glass edge space-y-2 rounded-xl p-3" onSubmit={e => { e.preventDefault(); if (photo) onStart(photo, prompt, extras) }}>
        {camera ? (
          <CameraCapture onCapture={camera === 'extra' ? b => addExtras([b]) : take} onClose={() => setCamera(null)} />
        ) : (
          <div
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={e => { e.preventDefault(); setDragging(false); drop(e.dataTransfer.files) }}
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
            <button type="button" onClick={() => setCamera('main')}
              className="press inline-flex min-h-tap items-center gap-1 rounded-sm bg-card px-2 py-1.5 text-body text-ink shadow-ring hover:shadow-ring-hover">
              <IconCamera className="h-3 w-3" /> Use camera
            </button>
          </div>
        )}

        {!camera && (
          <div className="space-y-1 rounded-md bg-card p-1 shadow-ring">
            <div className="flex items-center gap-1">
              <div className="min-w-0 flex-1">
                <p className="text-callout text-ink">More photos <span className="text-muted">· optional, up to {MAX_EXTRAS}</span></p>
                <p className="text-footnote text-muted">Other sides (left, right, back, top) or close-ups sharpen the 3D model and the identification.</p>
              </div>
              {extras.length < MAX_EXTRAS && (
                <>
                  <button type="button" onClick={() => extraFiles.current?.click()} aria-label="Add photos" title="Add photos"
                    className="press grid h-6 w-6 shrink-0 place-items-center rounded-sm text-ink hover:bg-ink/5"><IconPhoto className="h-3 w-3" /></button>
                  <button type="button" onClick={() => setCamera('extra')} aria-label="Take another photo" title="Take another photo"
                    className="press grid h-6 w-6 shrink-0 place-items-center rounded-sm text-ink hover:bg-ink/5"><IconCamera className="h-3 w-3" /></button>
                </>
              )}
            </div>
            {extras.length > 0 && (
              <ul className="flex flex-wrap gap-1.5 pt-1" aria-label="Extra photos">
                {extras.map((x, i) => <ExtraThumb key={i} photo={x} n={i + 2} onRemove={() => setExtras(prev => prev.filter((_, j) => j !== i))} />)}
              </ul>
            )}
            <input ref={extraFiles} type="file" accept="image/*" multiple className="sr-only" tabIndex={-1}
              onChange={e => { addExtras(e.target.files ?? []); e.target.value = '' }} />
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
