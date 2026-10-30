import { useEffect, useRef, useState } from 'react'
import { IconCamera, IconX } from './icons'

/** Live camera preview; capturing grabs one full-resolution frame as a JPEG. */
export function CameraCapture({ onCapture, onClose }: { onCapture: (photo: Blob) => void; onClose: () => void }) {
  const video = useRef<HTMLVideoElement>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let stream: MediaStream | null = null
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1920 } }, audio: false })
      .then(s => { stream = s; if (video.current) video.current.srcObject = s })
      .catch(e => setError(e instanceof Error ? e.message : String(e)))
    return () => stream?.getTracks().forEach(t => t.stop())
  }, [])

  const capture = () => {
    const v = video.current
    if (!v || !v.videoWidth) return
    const c = document.createElement('canvas')
    c.width = v.videoWidth; c.height = v.videoHeight
    c.getContext('2d')!.drawImage(v, 0, 0)
    c.toBlob(b => b && onCapture(b), 'image/jpeg', 0.92)
  }

  return (
    <div className="space-y-2">
      <div className="relative overflow-hidden rounded-md bg-black">
        <video ref={video} autoPlay playsInline muted className="aspect-video w-full object-cover" />
        {error && <p className="absolute inset-0 p-2 text-callout text-white">Camera unavailable: {error}</p>}
      </div>
      <div className="flex gap-1">
        <button type="button" onClick={capture} disabled={!!error}
          className="press inline-flex min-h-tap items-center gap-1 rounded-sm bg-accent-fill px-2 py-1.5 text-headline text-white hover:bg-accent-strong disabled:opacity-40">
          <IconCamera className="h-3 w-3" strokeWidth={2} /> Capture
        </button>
        <button type="button" onClick={onClose}
          className="press inline-flex min-h-tap items-center gap-1 rounded-sm px-2 py-1.5 text-body text-ink shadow-ring hover:shadow-ring-hover">
          <IconX className="h-3 w-3" /> Cancel
        </button>
      </div>
    </div>
  )
}
