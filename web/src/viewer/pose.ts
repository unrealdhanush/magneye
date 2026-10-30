import * as THREE from 'three'
import type { Anatomy, Vec3 } from '../types'

const DEG = Math.PI / 180
// Below this every angle counts as straight-on (matches FRONTAL_DEG in src/magneye/pose.py).
const FRONTAL_DEG = 12

function angles(anatomy: Anatomy): [number, number, number] {
  const c = anatomy.object.camera
  return [c?.azimuth_deg ?? 0, c?.elevation_deg ?? 0, c?.roll_deg ?? 0]
}

/**
 * Object frame -> photo frame, as in src/magneye/pose.py: the parts live in the object's own frame (x right, y up,
 * z out of its front) and the photo saw it from the camera's azimuth, elevation and roll. Rz(roll)·Rx(el)·Ry(-az).
 */
export function photoRotation(anatomy: Anatomy): THREE.Matrix4 {
  const [a, e, r] = angles(anatomy)
  return new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(e * DEG, -a * DEG, r * DEG, 'ZXY'))
}

export const isFrontal = (anatomy: Anatomy) => angles(anatomy).every(v => Math.abs(v) < FRONTAL_DEG)

/**
 * A photo taken from so far round, above or tilted that its outline, pushed back along the line of sight and
 * trimmed to the object's box, no longer resembles the object (an object lying on its side, shot from above).
 */
export function isSteep(anatomy: Anatomy) {
  const [a, e, r] = angles(anatomy)
  return Math.abs(r) > 30 || Math.abs(a) > 50 || Math.abs(e) > 55
}

/** How far an object-frame box of `size` spans along the photo's x, y and z. */
export function spanInPhoto(anatomy: Anatomy, size: Vec3): Vec3 {
  const m = photoRotation(anatomy).elements // column-major
  return [0, 1, 2].map(i => [0, 1, 2].reduce((sum, j) => sum + Math.abs(m[j * 4 + i]) * size[j], 0)) as Vec3
}

/** The object's span in the photo plane (x, y) and toward the camera (z), in millimetres. */
export const photoExtent = (anatomy: Anatomy) => spanInPhoto(anatomy, anatomy.object.dimensions_mm)

/**
 * Where the 3D camera starts: from the side the photo was taken, swung to a three-quarter view (34° around and
 * 19° up, the long-standing default for straight-on photos) so the parts read in depth.
 */
export function startCamera(anatomy: Anatomy, distance = 6.4): Vec3 {
  const [a, e] = angles(anatomy)
  const az = (a + 34) * DEG, el = Math.min(Math.max(e + 19, -30), 70) * DEG
  return [Math.sin(az) * Math.cos(el) * distance, Math.sin(el) * distance, Math.cos(az) * Math.cos(el) * distance]
}
