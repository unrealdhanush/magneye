// Sharing and export helpers: file names, downloads, clipboard, and the Blueprint as a standalone SVG or PNG.

/** A file-name-safe version of the object's name. */
export const fileSlug = (name: string) => name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'magneye'

export function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = filename
  document.body.appendChild(a); a.click(); a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 2000)
}

export const canvasToBlob = (c: HTMLCanvasElement, type = 'image/png') =>
  new Promise<Blob>((resolve, reject) => c.toBlob(b => (b ? resolve(b) : reject(new Error('The image could not be created.'))), type))

/** Copy text to the clipboard; falls back to a hidden textarea where the async API is blocked. */
export async function copyText(text: string): Promise<boolean> {
  try { await navigator.clipboard.writeText(text); return true } catch { /* fall through */ }
  const t = document.createElement('textarea')
  t.value = text; t.setAttribute('readonly', ''); t.style.cssText = 'position:fixed;opacity:0'
  document.body.appendChild(t); t.select()
  let ok = false
  try { ok = document.execCommand('copy') } catch { /* blocked */ }
  t.remove()
  return ok
}

// The sheet is styled by classes in index.css, which a saved file doesn't have, so each element's final style is written inline.
const INLINE = [
  'fill', 'fill-opacity', 'stroke', 'stroke-width', 'stroke-opacity', 'stroke-dasharray', 'stroke-linecap', 'stroke-linejoin', 'opacity',
  'font-family', 'font-size', 'font-weight', 'letter-spacing', 'text-anchor', 'dominant-baseline', 'font-variant-numeric',
]

/**
 * The Blueprint as a self-contained SVG string: as drawn, but finished (animations done) and without hover, selection or drag
 * highlights. Works on a copy, so the live sheet is untouched.
 */
export function blueprintSvg(source: SVGSVGElement): { svg: string; w: number; h: number } {
  const clone = source.cloneNode(true) as SVGSVGElement
  const host = document.createElement('div')
  host.style.cssText = 'position:fixed;left:-99999px;top:0;width:0;height:0;overflow:hidden;visibility:hidden;pointer-events:none'
  host.appendChild(clone)
  document.body.appendChild(host)
  try {
    const els = [clone, ...Array.from(clone.querySelectorAll<SVGElement>('*'))]
    for (const el of els) {
      el.classList.remove('is-hovered', 'is-selected', 'is-dragging')
      el.style.setProperty('animation', 'none')
      if (el.classList.contains('bp-fade')) el.style.setProperty('opacity', '1')
      if (el.classList.contains('bp-draw')) {
        el.style.setProperty('stroke-dashoffset', '0')
        if (!el.closest('.is-hidden')) el.style.setProperty('stroke-dasharray', 'none')
      }
    }
    for (const el of els) {
      const cs = getComputedStyle(el)
      for (const p of INLINE) el.style.setProperty(p, cs.getPropertyValue(p))
    }
    const vb = clone.viewBox.baseVal
    const w = vb.width || 1600, h = vb.height || 1000
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    clone.setAttribute('width', String(w)); clone.setAttribute('height', String(h))
    clone.removeAttribute('class'); clone.removeAttribute('role')
    return { svg: `<?xml version="1.0" encoding="UTF-8"?>\n${new XMLSerializer().serializeToString(clone)}`, w, h }
  } finally { host.remove() }
}

/** Rasterise an SVG string at the given scale. */
export async function svgToPng(svg: string, w: number, h: number, scale = 2): Promise<Blob> {
  const url = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml;charset=utf-8' }))
  try {
    const img = new Image()
    img.src = url
    await img.decode()
    const c = document.createElement('canvas')
    c.width = Math.round(w * scale); c.height = Math.round(h * scale)
    c.getContext('2d')!.drawImage(img, 0, 0, c.width, c.height)
    return await canvasToBlob(c)
  } finally { URL.revokeObjectURL(url) }
}
