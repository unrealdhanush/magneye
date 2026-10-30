import type { Anatomy, Answer, Ground, Session } from './types'

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init)
  if (!r.ok) {
    let detail = r.statusText
    try { detail = (await r.json()).detail ?? detail } catch { /* not JSON */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return r.json() as Promise<T>
}

const json = (body: unknown): RequestInit => ({
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
})

export const api = {
  health: () => call<{ ok: boolean; model: string; has_key: boolean; sam3d: boolean }>('/api/health'),
  createSession: (image: Blob, prompt: string) => {
    const form = new FormData()
    form.append('image', image, 'photo.jpg')
    form.append('prompt', prompt)
    return call<Pick<Session, 'id' | 'image_url' | 'size' | 'prompt'>>('/api/sessions', { method: 'POST', body: form })
  },
  session: (id: string) => call<Session>(`/api/sessions/${id}`),
  anatomy: (id: string) => call<Anatomy>(`/api/sessions/${id}/anatomy`, { method: 'POST' }),
  ground: (id: string) => call<{ anatomy: Anatomy; ground: Ground }>(`/api/sessions/${id}/ground`, { method: 'POST' }),
  reconstruct: (id: string) => call<{ mesh_url: string }>(`/api/sessions/${id}/reconstruct`, { method: 'POST' }),
  ask: (id: string, prompt: string) => call<Answer>(`/api/sessions/${id}/ask`, json({ prompt })),
  transcribe: (wav: Blob) => {
    const form = new FormData()
    form.append('audio', wav, 'clip.wav')
    return call<{ text: string }>('/api/transcribe', { method: 'POST', body: form })
  },
}
