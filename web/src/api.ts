import type { AgentEvent, Session } from './types'

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init)
  if (!r.ok) throw new Error(await detail(r))
  return r.json() as Promise<T>
}

async function detail(r: Response) {
  try { const d = (await r.json()).detail; return typeof d === 'string' ? d : JSON.stringify(d) } catch { return r.statusText }
}

/** POST and read the server-sent event stream an agent produces, calling `onEvent` for each event. */
async function stream(path: string, body: unknown, onEvent: (e: AgentEvent) => void): Promise<void> {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body ?? {}) })
  if (!r.ok || !r.body) throw new Error(await detail(r))
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value
    const chunks = buffer.split('\n\n')
    buffer = chunks.pop() ?? ''
    for (const chunk of chunks) {
      const data = chunk.split('\n').filter(l => l.startsWith('data:')).map(l => l.slice(5).trim()).join('')
      if (data) onEvent(JSON.parse(data) as AgentEvent)
    }
  }
}

export const api = {
  health: () => call<{ ok: boolean; model: string; has_key: boolean; sam3d: boolean }>('/api/health'),
  createSession: (image: Blob, prompt: string, extras: Blob[] = []) => {
    const form = new FormData()
    form.append('image', image, 'photo.jpg')
    form.append('prompt', prompt)
    extras.forEach((x, i) => form.append('extras', x, `extra-${i + 1}.jpg`))
    return call<Pick<Session, 'id' | 'image_url' | 'size' | 'prompt'>>('/api/sessions', { method: 'POST', body: form })
  },
  session: (id: string) => call<Session>(`/api/sessions/${id}`),
  build: (id: string, onEvent: (e: AgentEvent) => void) => stream(`/api/sessions/${id}/build`, {}, onEvent),
  ask: (id: string, prompt: string, onEvent: (e: AgentEvent) => void) => stream(`/api/sessions/${id}/ask`, { prompt }, onEvent),
  reconstruct: (id: string) => call<{ mesh_url: string }>(`/api/sessions/${id}/reconstruct`, { method: 'POST' }),
  transcribe: (wav: Blob) => {
    const form = new FormData()
    form.append('audio', wav, 'clip.wav')
    return call<{ text: string }>('/api/transcribe', { method: 'POST', body: form })
  },
}
