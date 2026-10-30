import { useEffect, useState } from 'react'
import * as THREE from 'three'
import { toCreasedNormals } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type { Anatomy, Vec3 } from '../types'

type Polygons = [number, number][][]
/** A modeled part's silhouettes in each blueprint view, sized and turned as placed, relative to the part's centre (mm). */
export interface PartOutline { min: Vec3; max: Vec3; views: Record<'front' | 'top' | 'right', Polygons> }
interface Encoded { positions: string; indices: string; min: Vec3; max: Vec3; outlines: PartOutline['views'] }
export interface PartModels { meshes: Map<string, THREE.BufferGeometry>; outlines: Map<string, PartOutline> }
const NONE: PartModels = { meshes: new Map(), outlines: new Map() }

const bytes = (b64: string) => Uint8Array.from(atob(b64), c => c.charCodeAt(0)).buffer

/**
 * A built part mesh in the part's own frame, scaled to unit size about the recipe's origin (the part's centre), so
 * the viewer can size it to the part's size_mm and turn it by its rotation like any other piece.
 */
function decode(m: Encoded): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(bytes(m.positions)), 3))
  g.setIndex(new THREE.BufferAttribute(new Uint32Array(bytes(m.indices)), 1))
  const span = m.max.map((v, i) => Math.max(v - m.min[i], 1e-6))
  g.scale(1 / span[0], 1 / span[1], 1 / span[2])
  // Smooth across gentle curves, crisp at real edges.
  return toCreasedNormals(g, Math.PI / 6)
}

/**
 * The part meshes Manifold built from the modeler's recipes, refetched whenever a recipe changes (keyed on the
 * recipes themselves, so moving or resizing a part doesn't refetch).
 */
export function usePartModels(sid: string, anatomy: Anatomy | null): PartModels {
  const key = anatomy ? JSON.stringify(anatomy.parts.filter(p => p.model).map(p => [p.id, p.model])) : '[]'
  const [models, setModels] = useState<PartModels>(NONE)
  useEffect(() => {
    if (key === '[]') { setModels(NONE); return }
    let alive = true
    fetch(`/api/sessions/${sid}/models`).then(r => (r.ok ? r.json() : null)).then((d: { parts: Record<string, Encoded> } | null) => {
      if (!alive || !d) return
      const entries = Object.entries(d.parts)
      setModels({
        meshes: new Map(entries.map(([id, m]) => [id, decode(m)])),
        outlines: new Map(entries.filter(([, m]) => m.outlines).map(([id, m]) => [id, { min: m.min, max: m.max, views: m.outlines }])),
      })
    }).catch(() => {})
    return () => { alive = false }
  }, [sid, key])
  return models
}
