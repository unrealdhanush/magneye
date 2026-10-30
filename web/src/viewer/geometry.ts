import * as THREE from 'three'
import { SVGLoader } from 'three/examples/jsm/loaders/SVGLoader.js'
import type { Anatomy, Ground, Part, Vec3 } from '../types'

/** World units per millimetre: the object's longest side becomes 2 units. */
export function worldScale(anatomy: Anatomy): number {
  return 2 / Math.max(...anatomy.object.dimensions_mm, 1)
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
export function partShape(part: Part, s: number): PartShape {
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
  const axis = roundAxis(size, part.shape === 'disc' || part.shape === 'ring' ? 'smallest' : 'unique')
  const len = size[axis]
  const radial = size.filter((_, i) => i !== axis) as [number, number]
  const diameter = (radial[0] + radial[1]) / 2
  // Radial scale keeps any ellipse the model asked for: first radial dim maps to local X, second to local Z.
  const ellipse: Vec3 = [radial[0] / diameter, 1, radial[1] / diameter]
  switch (part.shape) {
    case 'cylinder':
    case 'disc':
      return { geometry: shared('cyl', () => new THREE.CylinderGeometry(0.5, 0.5, 1, 40)), scale: [radial[0], len, radial[1]], axis: AXIS_ROT[axis] }
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

/** Where each part flies in the exploded view: outer layers travel farthest so the inside opens up. */
export function explodeDirection(part: Part, s: number, maxLayer: number): THREE.Vector3 {
  const c = new THREE.Vector3(...part.center_mm).multiplyScalar(s)
  const dir = new THREE.Vector3(c.x, c.y, c.z * 2.5)
  if (dir.lengthSq() < 1e-4) dir.set(0, 0, part.kind === 'exterior' ? 1 : 0.4)
  dir.normalize()
  const reach = 0.5 + 0.9 * (maxLayer - part.layer) / Math.max(maxLayer, 1)
  return dir.multiplyScalar(reach)
}

/**
 * The photographed silhouette (SAM 3.1 mask) extruded to Wonder's depth estimate, with the photo
 * projected onto the front and back faces. Returns null when grounding found no object.
 */
export function silhouetteShell(anatomy: Anatomy, ground: Ground): THREE.ExtrudeGeometry | null {
  const obj = ground.object
  if (!obj?.mask_svg) return null
  const [dw, dh, dd] = anatomy.object.dimensions_mm
  const s = worldScale(anatomy)
  const [left, top, right, bottom] = obj.box
  const [ml, mt, mr, mb] = obj.mask_bounds
  const [mw, mh] = obj.mask_size
  const [iw, ih] = ground.image_size
  const ow = right - left, oh = bottom - top

  // mask-local pixel -> image pixel -> object millimetres -> world units
  const toWorld = (u: number, v: number) => {
    const px = ml + u * ((mr - ml) / mw), py = mt + v * ((mb - mt) / mh)
    return new THREE.Vector2(((px - left) / ow - 0.5) * dw * s, (0.5 - (py - top) / oh) * dh * s)
  }
  const svg = new SVGLoader().parse(`<svg xmlns="http://www.w3.org/2000/svg"><path d="${obj.mask_svg}"/></svg>`)
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
  if (!shapes.length) return null
  // Keep the big pieces only: tiny islands are mask noise.
  const area = (sh: THREE.Shape) => Math.abs(THREE.ShapeUtils.area(sh.getPoints()))
  const biggest = Math.max(...shapes.map(area))
  const kept = shapes.filter(sh => area(sh) > biggest * 0.02)

  const depth = dd * s
  const bevel = Math.min(depth * 0.25, 0.08)
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
  const g = new THREE.ExtrudeGeometry(kept, {
    depth: Math.max(depth - 2 * bevel, 0.01), bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel * 0.6,
    bevelSegments: 5, curveSegments: 4, UVGenerator: uvGenerator,
  })
  g.translate(0, 0, -depth / 2 + bevel)
  g.computeVertexNormals()
  return g
}
