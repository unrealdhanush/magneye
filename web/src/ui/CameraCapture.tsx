import { useEffect, useRef, useState } from 'react'
import { IconCamera, IconX } from './icons'

const isPhone = (d: MediaDeviceInfo) => /iphone|ipad/i.test(d.label)
const STORE = 'magneye.camera'
const TIMER_STORE = 'magneye.camera-timer'
const TIMERS = [0, 3, 10]

// The camera the user last picked, kept per browser like a meeting app's camera setting.
function remembered(): string | null {
  try { return localStorage.getItem(STORE) } catch { return null }
}
function remember(id: string | null) {
  try { if (id) localStorage.setItem(STORE, id); else localStorage.removeItem(STORE) } catch { /* storage unavailable */ }
}
function rememberedTimer(): number {
  try { const n = Number(localStorage.getItem(TIMER_STORE)); return TIMERS.includes(n) ? n : 0 } catch { return 0 }
}

async function listCameras() {
  return (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput')
}

/** The camera to use when the user has not picked one this time: their last pick if it is here, else an iPhone. */
function preferred(list: MediaDeviceInfo[]): string | null {
  const last = remembered()
  return list.find(d => d.deviceId === last)?.deviceId ?? list.find(isPhone)?.deviceId ?? null
}

/**
 * Live camera preview; capturing grabs one full-resolution frame as a JPEG. Like a meeting app it lists every
 * camera, so an iPhone used through Continuity Camera can be picked, remembers the pick, and switches to the iPhone
 * when it shows up unless another camera was chosen.
 */
export function CameraCapture({ onCapture, onClose }: { onCapture: (photo: Blob) => void; onClose: () => void }) {
  const video = useRef<HTMLVideoElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([])
  // The camera asked for (null: the browser's default) and the one actually streaming.
  const [deviceId, setDeviceId] = useState<string | null>(remembered)
  const [current, setCurrent] = useState<string | null>(null)
  const live = useRef<string | null>(null)
  const [resolution, setResolution] = useState<string | null>(null)
  // Once the user picks a camera in this view, stop switching automatically.
  const chosen = useRef(false)
  // Self-timer: with an iPhone as the camera there is no shutter on the phone, so this frees both hands.
  const [timer, setTimer] = useState(rememberedTimer)
  const [countdown, setCountdown] = useState<number | null>(null)

  useEffect(() => {
    let stream: MediaStream | null = null
    let alive = true
    const size = { width: { ideal: 3840 }, height: { ideal: 2160 } }
    navigator.mediaDevices.getUserMedia({ video: deviceId ? { deviceId: { exact: deviceId }, ...size } : { facingMode: 'environment', ...size }, audio: false })
      .then(async s => {
        if (!alive) { s.getTracks().forEach(t => t.stop()); return }
        stream = s
        setError(null)
        if (video.current) video.current.srcObject = s
        const settings = s.getVideoTracks()[0]?.getSettings()
        setResolution(settings?.width ? `${settings.width} × ${settings.height}` : null)
        setCurrent(settings?.deviceId ?? null)
        live.current = settings?.deviceId ?? null
        // Device labels are only readable once camera access is granted, so list cameras after opening one.
        const list = await listCameras()
        if (!alive) return
        setCameras(list)
        const want = chosen.current ? null : preferred(list)
        if (want && want !== settings?.deviceId) setDeviceId(want)
      })
      .catch(e => {
        if (!alive) return
        // A remembered camera that is gone (an iPhone out of range) falls back to the default one.
        if (deviceId) { remember(null); setDeviceId(null); return }
        setError(e instanceof Error ? e.message : String(e))
      })
    return () => { alive = false; stream?.getTracks().forEach(t => t.stop()) }
  }, [deviceId])

  // Cameras come and go (an iPhone woken or carried off): refresh the list, follow the preferred camera, and fall
  // back to the default if the live one disappeared.
  useEffect(() => {
    const onChange = async () => {
      const list = await listCameras()
      setCameras(list)
      const want = chosen.current ? null : preferred(list)
      if (want && want === live.current) return
      setDeviceId(id => want ?? (id && list.some(d => d.deviceId === id) ? id : null))
    }
    navigator.mediaDevices.addEventListener('devicechange', onChange)
    return () => navigator.mediaDevices.removeEventListener('devicechange', onChange)
  }, [])

  const pick = (id: string) => { chosen.current = true; remember(id); setDeviceId(id) }

  const grab = () => {
    const v = video.current
    if (!v || !v.videoWidth) return
    const c = document.createElement('canvas')
    c.width = v.videoWidth; c.height = v.videoHeight
    c.getContext('2d')!.drawImage(v, 0, 0)
    c.toBlob(b => b && onCapture(b), 'image/jpeg', 0.92)
  }

  useEffect(() => {
    if (countdown === null) return
    if (countdown === 0) { setCountdown(null); grab(); return }
    const id = setTimeout(() => setCountdown(n => (n === null ? null : n - 1)), 1000)
    return () => clearTimeout(id)
  }, [countdown]) // eslint-disable-line react-hooks/exhaustive-deps

  // Capture again while counting down cancels the timer.
  const capture = () => {
    if (countdown !== null) setCountdown(null)
    else if (timer) setCountdown(timer)
    else grab()
  }
  const chooseTimer = (n: number) => {
    setTimer(n)
    try { localStorage.setItem(TIMER_STORE, String(n)) } catch { /* storage unavailable */ }
  }

  const hasPhone = cameras.some(isPhone)
  return (
    <div className="space-y-2">
      <div className="relative overflow-hidden rounded-md bg-black">
        <video ref={video} autoPlay playsInline muted className="aspect-video w-full object-cover" />
        {error && <p className="absolute inset-0 p-2 text-callout text-white">Camera unavailable: {error}</p>}
        {resolution && <span className="absolute bottom-1 left-1 rounded-sm bg-black/60 px-1 font-mono text-caption2 leading-5 text-white">{resolution}</span>}
        {countdown !== null && (
          <span className="absolute inset-0 grid place-items-center bg-black/30">
            <span role="timer" aria-live="assertive"
              className="glass grid h-8 w-8 place-items-center rounded-full font-display text-largeTitle text-white tabular">
              {countdown}
            </span>
          </span>
        )}
      </div>
      {cameras.length > 0 && (
        <label className="flex items-center gap-1 text-subheadline text-ink">
          Camera
          <select value={current ?? deviceId ?? ''} onChange={e => pick(e.target.value)}
            className="min-h-tap min-w-0 flex-1 rounded-sm bg-card px-1 text-body text-ink shadow-ring">
            {cameras.map((c, i) => <option key={c.deviceId} value={c.deviceId}>{c.label || `Camera ${i + 1}`}</option>)}
          </select>
        </label>
      )}
      {!hasPhone && cameras.length > 0 && (
        <p className="text-footnote text-muted">
          For a sharper shot, use your iPhone as the camera (Continuity Camera): keep it near this Mac, signed in to
          the same Apple ID with Wi-Fi and Bluetooth on, and it appears in this list.
        </p>
      )}
      <div className="flex flex-wrap items-center gap-1">
        <button type="button" onClick={capture} disabled={!!error}
          className="press inline-flex min-h-tap items-center gap-1 rounded-sm bg-accent-fill px-2 py-1.5 text-headline text-white hover:bg-accent-strong disabled:opacity-40">
          <IconCamera className="h-3 w-3" strokeWidth={2} /> {countdown !== null ? 'Cancel timer' : timer ? `Capture in ${timer} s` : 'Capture'}
        </button>
        <div role="radiogroup" aria-label="Self-timer" className="inline-flex rounded-sm bg-card p-px shadow-ring">
          {TIMERS.map(n => (
            <button key={n} type="button" role="radio" aria-checked={timer === n} onClick={() => chooseTimer(n)}
              className={`press min-h-tap rounded-sm px-1.5 text-subheadline ${timer === n ? 'bg-accent-fill text-white' : 'text-ink hover:bg-ink/5'}`}>
              {n ? `${n} s` : 'No timer'}
            </button>
          ))}
        </div>
        <button type="button" onClick={onClose}
          className="press inline-flex min-h-tap items-center gap-1 rounded-sm px-2 py-1.5 text-body text-ink shadow-ring hover:shadow-ring-hover">
          <IconX className="h-3 w-3" /> Cancel
        </button>
      </div>
    </div>
  )
}
