import * as THREE from 'three'
import { SVGLoader } from 'three/examples/jsm/loaders/SVGLoader.js'
import type { Anatomy, Ground, MaskMatch, Part, Vec3 } from '../types'

/**
 * The object's main body in millimetres: every part except accessories (a bracelet, a long cable, a stand) when they
 * make the object much bigger than its body, which would otherwise shrink the interesting part to a corner of the
 * view. Otherwise it is the whole object.
 */
export function bodyBox(anatomy: Anatomy): { center: Vec3; size: Vec3; trimmed: boolean } {
  const whole = { center: [0, 0, 0] as Vec3, size: anatomy.object.dimensions_mm, trimmed: false }
  const body = anatomy.parts.filter(p => p.kind !== 'accessory')
  if (!body.length || body.length === anatomy.parts.length) return whole
  const span = (parts: Part[]) => {
    const lo = [0, 1, 2].map(i => Math.min(...parts.map(p => p.center_mm[i] - p.size_mm[i] / 2)))
    const hi = [0, 1, 2].map(i => Math.max(...parts.map(p => p.center_mm[i] + p.size_mm[i] / 2)))
    return { center: lo.map((v, i) => (v + hi[i]) / 2) as Vec3, size: lo.map((v, i) => hi[i] - v) as Vec3 }
  }
  const own = span(body), all = span(anatomy.parts)
  // Compare against all parts (the same estimates), across the photo plane where accessories hang off.
  if (own.size[0] >= 0.75 * all.size[0] && own.size[1] >= 0.75 * all.size[1]) return whole
  return { ...own, trimmed: true }
}

/** World units per millimetre: the main body's longest side becomes 2 units. */
export function worldScale(anatomy: Anatomy): number {
  return 2 / Math.max(...bodyBox(anatomy).size, 1)
}

export const DEG = Math.PI / 180

/** Index of the dimension that differs most from the other two: the axis of a round part. */
function roundAxis(size: Vec3, prefer: 'unique' | 'smallest'): 0 | 1 | 2 {
  if (prefer === 'smallest') return size.indexOf(Math.min(...size)) as 0 | 1 | 2
  const score = size.map((d, i) => {
    const others = size.filter((_, j) => j !== i)
    return Math.abs(d - (others[0] + others[1]) / 2) - Math.abs(others[0] - others[1])
  })
  return score.indexOf(Math.max(...score)) as 0 | 1 | 2
}

/** Rotation that turns a Y-axis primitive onto the given axis. */
const AXIS_ROT: Record<0 | 1 | 2, THREE.Euler> = {
  0: new THREE.Euler(0, 0, -Math.PI / 2),
  1: new THREE.Euler(0, 0, 0),
  2: new THREE.Euler(Math.PI / 2, 0, 0),
}

export interface PartShape {
  geometry: THREE.BufferGeometry
  /** Scale applied to the geometry in its Y-axis frame. */
  scale: Vec3
  /** Rotation from the Y-axis frame onto the part's own axis. */
  axis: THREE.Euler
}

const cache = new Map<string, THREE.BufferGeometry>()
function shared(key: string, make: () => THREE.BufferGeometry) {
  if (!cache.has(key)) cache.set(key, make())
  return cache.get(key)!
}

function helix(radius: number, length: number, tube: number) {
  const turns = 7
  const pts = Array.from({ length: turns * 24 + 1 }, (_, i) => {
    const a = (i / 24) * Math.PI * 2
    return new THREE.Vector3(Math.cos(a) * radius, -length / 2 + (i / (turns * 24)) * length, Math.sin(a) * radius)
  })
  return new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), turns * 32, tube, 8, false)
}

/** Primitive geometry for a part, sized in world units. */
export function partShape(part: Pick<Part, 'shape' | 'size_mm'>, s: number): PartShape {
  const size = part.size_mm.map(v => Math.max(v, 0.5) * s) as Vec3
  const ident = new THREE.Euler()
  switch (part.shape) {
    case 'box':
    case 'plate':
      return { geometry: shared('box', () => new THREE.BoxGeometry(1, 1, 1)), scale: size, axis: ident }
    case 'rounded_box':
      return { geometry: shared('rbox', () => roundedBox()), scale: size, axis: ident }
    case 'sphere':
      return { geometry: shared('sphere', () => new THREE.SphereGeometry(0.5, 32, 16)), scale: size, axis: ident }
    default:
      break
  }
  const flat = part.shape === 'disc' || part.shape === 'ring' || part.shape === 'gear' || part.shape === 'spiral'
  const axis = roundAxis(size, flat ? 'smallest' : 'unique')
  const len = size[axis]
  const radial = size.filter((_, i) => i !== axis) as [number, number]
  const diameter = (radial[0] + radial[1]) / 2
  // Radial scale keeps any ellipse the model asked for: first radial dim maps to local X, second to local Z.
  const ellipse: Vec3 = [radial[0] / diameter, 1, radial[1] / diameter]
  switch (part.shape) {
    case 'cylinder':
    case 'disc':
      return { geometry: shared('cyl', () => new THREE.CylinderGeometry(0.5, 0.5, 1, 40)), scale: [radial[0], len, radial[1]], axis: AXIS_ROT[axis] }
    case 'gear':
      return { geometry: shared('gear', () => gear()), scale: [radial[0], len, radial[1]], axis: AXIS_ROT[axis] }
    case 'spiral':
      // The spring's wire stays thin; only its footprint follows the part's size.
      return { geometry: shared('spiral', () => spiral()), scale: [radial[0], 1, radial[1]], axis: AXIS_ROT[axis] }
    case 'cone':
      return { geometry: shared('cone', () => new THREE.ConeGeometry(0.5, 1, 40)), scale: [radial[0], len, radial[1]], axis: AXIS_ROT[axis] }
    case 'capsule': {
      const r = diameter / 2
      const g = new THREE.CapsuleGeometry(r, Math.max(len - 2 * r, 0.001), 8, 24)
      return { geometry: g, scale: ellipse, axis: AXIS_ROT[axis] }
    }
    case 'ring': {
      const tube = Math.max(Math.min(len / 2, diameter * 0.12), 0.004)
      const g = new THREE.TorusGeometry(diameter / 2 - tube, tube, 16, 64).rotateX(Math.PI / 2)
      return { geometry: g, scale: ellipse, axis: AXIS_ROT[axis] }
    }
    case 'coil': {
      const tube = Math.max(diameter * 0.06, 0.003)
      return { geometry: helix(diameter / 2 - tube, len, tube), scale: ellipse, axis: AXIS_ROT[axis] }
    }
    default:
      return { geometry: shared('box', () => new THREE.BoxGeometry(1, 1, 1)), scale: size, axis: ident }
  }
}

/** A unit-diameter spur gear, axis along Y, unit thickness. */
function gear(teeth = 18) {
  const shape = new THREE.Shape()
  const rOut = 0.5, rIn = 0.42, step = (Math.PI * 2) / teeth
  for (let i = 0; i < teeth; i++) {
    const a = i * step
    const pts: [number, number][] = [[rIn, a], [rOut, a + step * 0.15], [rOut, a + step * 0.45], [rIn, a + step * 0.6]]
    pts.forEach(([r, t], k) => (i === 0 && k === 0 ? shape.moveTo : shape.lineTo).call(shape, Math.cos(t) * r, Math.sin(t) * r))
  }
  shape.closePath()
  const hub = new THREE.Path()
  hub.absarc(0, 0, 0.08, 0, Math.PI * 2, true)
  shape.holes.push(hub)
  const g = new THREE.ExtrudeGeometry(shape, { depth: 1, bevelEnabled: false, curveSegments: 4 })
  g.translate(0, 0, -0.5)
  g.rotateX(Math.PI / 2)
  return g
}

/** A unit-diameter flat spiral spring (hairspring, mainspring), axis along Y. */
function spiral(turns = 7) {
  const pts = Array.from({ length: turns * 48 + 1 }, (_, i) => {
    const t = i / (turns * 48), a = t * turns * Math.PI * 2, r = 0.06 + t * 0.44
    return new THREE.Vector3(Math.cos(a) * r, 0, Math.sin(a) * r)
  })
  return new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), turns * 64, 0.012, 6, false)
}

function roundedBox() {
  // A unit box with softened edges, built once and scaled per part.
  const shape = new THREE.Shape()
  const r = 0.18, h = 0.5
  shape.moveTo(-h + r, -h)
  shape.lineTo(h - r, -h); shape.quadraticCurveTo(h, -h, h, -h + r)
  shape.lineTo(h, h - r); shape.quadraticCurveTo(h, h, h - r, h)
  shape.lineTo(-h + r, h); shape.quadraticCurveTo(-h, h, -h, h - r)
  shape.lineTo(-h, -h + r); shape.quadraticCurveTo(-h, -h, -h + r, -h)
  const g = new THREE.ExtrudeGeometry(shape, { depth: 0.8, bevelEnabled: true, bevelSize: 0.1, bevelThickness: 0.1, bevelSegments: 4, curveSegments: 6 })
  g.center()
  return g
}

/**
 * Where each part flies in the exploded view: outer layers travel farthest so the inside opens up. Directions
 * radiate from the body's centre; accessories slide aside within the photo plane to clear the view.
 */
export function explodeDirection(part: Part, s: number, maxLayer: number, bodyCenter: THREE.Vector3): THREE.Vector3 {
  const c = new THREE.Vector3(...part.center_mm).multiplyScalar(s).sub(bodyCenter)
  if (part.kind === 'accessory') {
    c.z = 0
    if (c.lengthSq() < 1e-4) c.set(0, -1, 0)
    return c.normalize().multiplyScalar(1.6)
  }
  const dir = new THREE.Vector3(c.x, c.y, c.z * 2.5)
  if (dir.lengthSq() < 1e-4) dir.set(0, 0, part.kind === 'exterior' ? 1 : 0.4)
  dir.normalize()
  const reach = 0.5 + 0.9 * (maxLayer - part.layer) / Math.max(maxLayer, 1)
  return dir.multiplyScalar(reach)
}

/** A SAM 3.1 mask outline as shapes in world units, in the object's frame (photo = front view). */
export function maskShapes(match: MaskMatch, anatomy: Anatomy, ground: Ground): THREE.Shape[] {
  const obj = ground.object
  if (!obj || !match.mask_svg) return []
  const [dw, dh] = anatomy.object.dimensions_mm
  const s = worldScale(anatomy)
  const [left, top, right, bottom] = obj.box
  const [ml, mt, mr, mb] = match.mask_bounds
  const [mw, mh] = match.mask_size
  const ow = right - left, oh = bottom - top
  // mask-local pixel -> image pixel -> object millimetres -> world units
  const toWorld = (u: number, v: number) => {
    const px = ml + u * ((mr - ml) / mw), py = mt + v * ((mb - mt) / mh)
    return new THREE.Vector2(((px - left) / ow - 0.5) * dw * s, (0.5 - (py - top) / oh) * dh * s)
  }
  const svg = new SVGLoader().parse(`<svg xmlns="http://www.w3.org/2000/svg"><path d="${match.mask_svg}"/></svg>`)
  const shapes: THREE.Shape[] = []
  for (const path of svg.paths) {
    for (const shape of SVGLoader.createShapes(path)) {
      const { shape: outer, holes } = shape.extractPoints(4)
      if (outer.length < 3) continue
      const mapped = new THREE.Shape(outer.map(p => toWorld(p.x, p.y)))
      mapped.holes = holes.map(h => new THREE.Path(h.map(p => toWorld(p.x, p.y))))
      shapes.push(mapped)
    }
  }
  // Keep the big pieces only: tiny islands are mask noise.
  const area = (sh: THREE.Shape) => Math.abs(THREE.ShapeUtils.area(sh.getPoints()))
  const biggest = Math.max(0, ...shapes.map(area))
  return shapes.filter(sh => area(sh) > biggest * 0.02)
}

/** Material slots of a mask extrusion. */
export const FACE = { photo: 0, side: 1, cut: 2, back: 3 } as const

/**
 * Extrude mask shapes between two depths (world units), mapping the photo onto the faces.
 * Lids get their own material slots: `lids[0]` for the face at zFrom, `lids[1]` for the face at zTo.
 */
export function extrudeMask(shapes: THREE.Shape[], anatomy: Anatomy, ground: Ground, zFrom: number, zTo: number,
  lids: [number, number]): THREE.ExtrudeGeometry | null {
  const obj = ground.object
  if (!obj || !shapes.length) return null
  const [dw, dh] = anatomy.object.dimensions_mm
  const s = worldScale(anatomy)
  const [left, top, right, bottom] = obj.box
  const [iw, ih] = ground.image_size
  const ow = right - left, oh = bottom - top
  const toUV = (x: number, y: number) => {
    const px = left + (x / (dw * s) + 0.5) * ow, py = top + (0.5 - y / (dh * s)) * oh
    return new THREE.Vector2(px / iw, 1 - py / ih)
  }
  const uvGenerator = {
    generateTopUV: (_g: THREE.ExtrudeGeometry, v: number[], a: number, b: number, c: number) =>
      [a, b, c].map(i => toUV(v[i * 3], v[i * 3 + 1])),
    generateSideWallUV: (_g: THREE.ExtrudeGeometry, v: number[], a: number, b: number, c: number, d: number) =>
      [a, b, c, d].map(i => toUV(v[i * 3], v[i * 3 + 1])),
  }
  const depth = Math.max(zTo - zFrom, 0.004)
  const bevel = Math.min(depth * 0.2, 0.04)
  const g = new THREE.ExtrudeGeometry(shapes, {
    depth: Math.max(depth - 2 * bevel, 0.002), bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel * 0.5,
    bevelSegments: 4, curveSegments: 4, UVGenerator: uvGenerator,
  })
  // three.js puts both lids in one group per shape (the zFrom lid first, then the zTo lid); split them.
  const groups = g.groups.map(gr => ({ ...gr }))
  g.clearGroups()
  for (const gr of groups) {
    if (gr.materialIndex === 0) {
      const half = gr.count / 2
      g.addGroup(gr.start, half, lids[0])
      g.addGroup(gr.start + half, half, lids[1])
    } else g.addGroup(gr.start, gr.count, FACE.side)
  }
  g.translate(0, 0, zFrom + bevel)
  g.computeVertexNormals()
  return g
}

/** Average photo colour inside a mask: used for side walls and cut faces so they match the object. */
export function maskColor(image: CanvasImageSource & { width: number; height: number }, match: MaskMatch): THREE.Color {
  const [ml, mt, mr, mb] = match.mask_bounds
  const w = Math.max(1, Math.round(mr - ml)), h = Math.max(1, Math.round(mb - mt))
  const scale = Math.min(1, 96 / Math.max(w, h))
  const cw = Math.max(1, Math.round(w * scale)), ch = Math.max(1, Math.round(h * scale))
  const canvas = document.createElement('canvas')
  canvas.width = cw; canvas.height = ch
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) return new THREE.Color('#8e8e93')
  ctx.save()
  ctx.scale(cw / match.mask_size[0], ch / match.mask_size[1])
  ctx.clip(new Path2D(match.mask_svg))
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.drawImage(image, ml, mt, mr - ml, mb - mt, 0, 0, cw, ch)
  ctx.restore()
  const px = ctx.getImageData(0, 0, cw, ch).data
  let r = 0, g = 0, b = 0, n = 0
  for (let i = 0; i < px.length; i += 4) {
    if (px[i + 3] < 128) continue
    r += px[i]; g += px[i + 1]; b += px[i + 2]; n++
  }
  return n ? new THREE.Color().setRGB(r / n / 255, g / n / 255, b / n / 255, THREE.SRGBColorSpace) : new THREE.Color('#8e8e93')
}

let pcb: THREE.CanvasTexture | null = null
/** A procedural circuit-board surface (traces, pads, vias), generated once. */
export function pcbTexture(): THREE.CanvasTexture {
  if (pcb) return pcb
  const c = document.createElement('canvas')
  c.width = c.height = 256
  const ctx = c.getContext('2d')!
  ctx.fillStyle = '#16794a'; ctx.fillRect(0, 0, 256, 256)
  let seed = 7
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
  ctx.strokeStyle = '#c9a54a'; ctx.lineWidth = 2; ctx.lineCap = 'round'
  for (let i = 0; i < 40; i++) {
    let x = rnd() * 256, y = rnd() * 256
    ctx.beginPath(); ctx.moveTo(x, y)
    for (let k = 0; k < 3; k++) {
      if (rnd() < 0.5) x += (rnd() - 0.5) * 120; else y += (rnd() - 0.5) * 120
      ctx.lineTo(x, y)
    }
    ctx.stroke()
    ctx.fillStyle = '#e3c26a'; ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill()
  }
  ctx.fillStyle = '#1b1b1d'
  for (let i = 0; i < 6; i++) ctx.fillRect(rnd() * 220, rnd() * 220, 18 + rnd() * 20, 12 + rnd() * 14)
  pcb = new THREE.CanvasTexture(c)
  pcb.colorSpace = THREE.SRGBColorSpace
  pcb.wrapS = pcb.wrapT = THREE.RepeatWrapping
  return pcb
}
