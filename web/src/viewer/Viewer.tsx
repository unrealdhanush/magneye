import { Bounds, ContactShadows, Html, Line, OrbitControls, useGLTF, useTexture } from '@react-three/drei'
import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber'
import { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import type { Anatomy, Ground, Material, Part, Playback, Scenario, Vec3 } from '../types'
import { bodyBox, DEG, explodeDirection, extrudeMask, FACE, maskColor, maskShapes, partShape, pcbTexture, worldScale } from './geometry'
import { ACCENT, CULPRIT, EFFECT_GLOW, MATERIAL_LOOK } from './palette'

export const REMOVE_MS = 750

export interface ViewerProps {
  anatomy: Anatomy
  ground: Ground | null
  imageUrl: string
  meshUrl: string | null
  explode: number
  xray: boolean
  playback: Playback
  focusIds: string[]
  selectedId: string | null
  onSelect: (id: string | null) => void
  /** When Assemble was pressed: parts fly home one by one, innermost first, and the shell closes last. */
  assembleAt: number | null
}

/** Per-part delay between arrivals during assembly; capped so the whole build-up stays around a second. */
const ASSEMBLE_TOTAL_MS = 900
const ARRIVAL_FLASH_MS = 320

/** Registry of each part's moving group, so pulses and labels can find parts in world space. */
type Registry = Map<string, THREE.Object3D>

export function Viewer(props: ViewerProps) {
  const floor = -(props.anatomy.object.dimensions_mm[1] * worldScale(props.anatomy)) / 2 - 0.5
  return (
    <Canvas camera={{ position: [3.4, 2.1, 5], fov: 38 }} dpr={[1, 2]} gl={{ antialias: true, alpha: true }}
      onPointerMissed={() => props.onSelect(null)}>
      <StudioEnvironment />
      <hemisphereLight args={['#ffffff', '#445566', 0.35]} />
      <directionalLight position={[3, 5, 4]} intensity={1.1} />
      <directionalLight position={[-4, -2, -3]} intensity={0.35} color="#88aaff" />
      <ContactShadows position={[0, floor, 0]} scale={9} blur={2.6} far={4} opacity={0.5} resolution={512} />
      {/* Frame the assembled main body once; explode/playback then move parts within that view. */}
      <Bounds fit clip margin={1.9}>
        <BodyFrame anatomy={props.anatomy} />
      </Bounds>
      <Suspense fallback={null}>
        <Scene {...props} />
      </Suspense>
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} minDistance={1.2} maxDistance={12} />
    </Canvas>
  )
}

/** An invisible box around the main body, so the camera frames it rather than any straps or cables. */
function BodyFrame({ anatomy }: { anatomy: Anatomy }) {
  const s = worldScale(anatomy)
  const { center, size } = bodyBox(anatomy)
  return (
    <mesh visible={false} position={center.map(v => v * s) as Vec3} raycast={() => null}>
      <boxGeometry args={size.map(v => v * s) as Vec3} />
    </mesh>
  )
}

/** Soft studio reflections generated on the GPU (no HDR download), so metal and plastic read correctly. */
function StudioEnvironment() {
  const gl = useThree(st => st.gl)
  const scene = useThree(st => st.scene)
  useEffect(() => {
    const pmrem = new THREE.PMREMGenerator(gl)
    const env = pmrem.fromScene(new RoomEnvironment(), 0.04).texture
    scene.environment = env
    return () => { scene.environment = null; env.dispose(); pmrem.dispose() }
  }, [gl, scene])
  return null
}

function Scene(p: ViewerProps) {
  const s = worldScale(p.anatomy)
  const registry = useMemo<Registry>(() => new Map(), [])
  const maxLayer = Math.max(0, ...p.anatomy.parts.map(x => x.layer))
  const ordered = useMemo(() => [...p.anatomy.parts].sort((a, b) => a.layer - b.layer), [p.anatomy.parts])
  // Assembly order: deepest layer first, and within a layer the parts nearest the centre first.
  const { rank, stagger } = useMemo(() => {
    const dist = (x: Part) => Math.hypot(...x.center_mm)
    const order = [...p.anatomy.parts].sort((a, b) => b.layer - a.layer || dist(a) - dist(b))
    return { rank: new Map(order.map((x, i) => [x.id, i])), stagger: Math.min(55, ASSEMBLE_TOTAL_MS / Math.max(order.length, 1)) }
  }, [p.anatomy.parts])
  const shellDelay = p.anatomy.parts.length * stagger

  return (
    <group>
      {ordered.map(part => (
        <PartMesh key={part.id} part={part} s={s} maxLayer={maxLayer} registry={registry} delay={(rank.get(part.id) ?? 0) * stagger} {...p} />
      ))}
      {p.ground && (p.meshUrl
        ? <MeshShell url={p.meshUrl} anatomy={p.anatomy} explode={p.explode} xray={p.xray} playback={p.playback} assembleAt={p.assembleAt} delay={shellDelay} />
        : <PhotoShell anatomy={p.anatomy} ground={p.ground} imageUrl={p.imageUrl} explode={p.explode} xray={p.xray} playback={p.playback} assembleAt={p.assembleAt} delay={shellDelay} />)}
      {p.playback.kind === 'scenario' && <ScenarioPath scenario={p.playback.scenario} startedAt={p.playback.startedAt} registry={registry} />}
      <CameraDirector playback={p.playback} registry={registry} anatomy={p.anatomy} s={s} />
    </group>
  )
}

type OrbitLike = { target: THREE.Vector3; update: () => void }
interface Flight { fromPos: THREE.Vector3; fromTarget: THREE.Vector3; toPos: THREE.Vector3; toTarget: THREE.Vector3; start: number }
const FLIGHT_MS = 1100
const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)

/** Flies the camera to a failure's culprit once the parts in the way have lifted off, and back home afterwards.
 *  It only drives during the flight, so the user can orbit freely before and after. */
function CameraDirector({ playback, registry, anatomy, s }: { playback: Playback; registry: Registry; anatomy: Anatomy; s: number }) {
  const camera = useThree(st => st.camera)
  const controls = useThree(st => st.controls) as unknown as OrbitLike | null
  const home = useRef<{ pos: THREE.Vector3; target: THREE.Vector3 } | null>(null)
  const flight = useRef<Flight | null>(null)

  useEffect(() => {
    if (!controls) return
    const fly = (toPos: THREE.Vector3, toTarget: THREE.Vector3) => {
      flight.current = { fromPos: camera.position.clone(), fromTarget: controls.target.clone(), toPos, toTarget, start: performance.now() }
    }
    if (playback.kind === 'failure') {
      if (!home.current) home.current = { pos: camera.position.clone(), target: controls.target.clone() }
      const culprit = anatomy.parts.find(p => p.id === playback.failure.culprit_part_id)
      const wait = playback.failure.access_path.length * REMOVE_MS + 150
      const timer = setTimeout(() => {
        const obj = registry.get(playback.failure.culprit_part_id)
        if (!obj || !culprit) return
        const target = obj.getWorldPosition(new THREE.Vector3())
        // Look in along the culprit's way out (the side its access path lifted off), blended with the current
        // view so the move feels continuous; come in close enough that it fills a good part of the view.
        const view = camera.position.clone().sub(controls.target).normalize()
        const out = target.clone().setZ(target.z + 0.5).normalize()
        const dir = view.add(out.multiplyScalar(1.5)).normalize()
        const size = Math.max(...culprit.size_mm) * s
        fly(target.clone().addScaledVector(dir, Math.min(Math.max(size * 12, 1.4), 2.6)), target)
      }, wait)
      return () => clearTimeout(timer)
    }
    if (home.current) {
      fly(home.current.pos, home.current.target)
      home.current = null
    }
  }, [playback, controls, camera, registry, anatomy, s])

  useFrame(() => {
    const f = flight.current
    if (!f || !controls) return
    const t = clamp01((performance.now() - f.start) / FLIGHT_MS), e = easeInOut(t)
    camera.position.lerpVectors(f.fromPos, f.toPos, e)
    controls.target.lerpVectors(f.fromTarget, f.toTarget, e)
    controls.update()
    if (t >= 1) flight.current = null
  })
  return null
}

/** Which step of a scenario is playing at time `t`, and how far through it (0..1). */
export function scenarioCursor(scenario: Scenario, elapsed: number) {
  const total = scenario.steps.reduce((sum, st) => sum + st.duration_ms, 0)
  if (elapsed >= total) return { index: scenario.steps.length - 1, progress: 1, done: true }
  let acc = 0
  for (let i = 0; i < scenario.steps.length; i++) {
    const d = scenario.steps[i].duration_ms
    if (elapsed < acc + d) return { index: i, progress: (elapsed - acc) / d, done: false }
    acc += d
  }
  return { index: scenario.steps.length - 1, progress: 1, done: true }
}

const clamp01 = (x: number) => Math.min(1, Math.max(0, x))

function surface(material: Material): THREE.MeshStandardMaterial {
  const look = MATERIAL_LOOK[material] ?? MATERIAL_LOOK.other
  const m = new THREE.MeshStandardMaterial({
    color: look.color, metalness: look.metalness, roughness: look.roughness, emissive: ACCENT, emissiveIntensity: 0, transparent: true,
  })
  if (material === 'pcb') { m.map = pcbTexture(); m.color.set('#ffffff') }
  if (material === 'glass' || material === 'fluid') m.opacity = 0.6
  return m
}

/** One drawable piece of a part, positioned relative to the part's centre. */
interface Piece { geometry: THREE.BufferGeometry; scale: Vec3; axis: THREE.Euler; position: Vec3; rotation: Vec3; material: THREE.Material | THREE.Material[] }

/**
 * How a part is drawn, best source first: its own SAM 3.1 outline extruded with the photo on its face
 * (visible parts), then the model's multi-primitive description, then its single primitive.
 */
function usePieces(part: Part, s: number, anatomy: Anatomy, ground: Ground | null, texture: THREE.Texture) {
  return useMemo(() => {
    const match = part.grounded ? ground?.parts[part.id] : undefined
    const home = new THREE.Vector3(...part.center_mm).multiplyScalar(s)
    if (match && ground) {
      const z0 = (part.center_mm[2] - part.size_mm[2] / 2) * s, z1 = (part.center_mm[2] + part.size_mm[2] / 2) * s
      const g = extrudeMask(maskShapes(match, anatomy, ground), anatomy, ground, z0, z1, [FACE.back, FACE.photo])
      if (g) {
        g.translate(-home.x, -home.y, -home.z)
        const tone = texture.image ? maskColor(texture.image as HTMLImageElement, match) : new THREE.Color('#3a3a3c')
        const mats = [
          new THREE.MeshStandardMaterial({ map: texture, roughness: 0.45, emissive: ACCENT, emissiveIntensity: 0, transparent: true }),
          new THREE.MeshStandardMaterial({ color: tone, roughness: 0.5, emissive: ACCENT, emissiveIntensity: 0, transparent: true }),
          new THREE.MeshStandardMaterial({ color: tone.clone().multiplyScalar(0.5), roughness: 0.8, emissive: ACCENT, emissiveIntensity: 0, transparent: true }),
          new THREE.MeshStandardMaterial({ color: tone, roughness: 0.5, emissive: ACCENT, emissiveIntensity: 0, transparent: true }),
        ]
        return { photo: true, pieces: [{ geometry: g, scale: [1, 1, 1] as Vec3, axis: new THREE.Euler(), position: [0, 0, 0] as Vec3, rotation: [0, 0, 0] as Vec3, material: mats }], materials: mats }
      }
    }
    const comps = (part.components ?? []).filter(c => c.size_mm?.length === 3 && c.offset_mm?.length === 3)
    const specs = comps.length
      ? comps.map(c => ({ shape: c.shape, size_mm: c.size_mm, material: c.material ?? part.material, offset: c.offset_mm, rot: c.rotation_deg ?? [0, 0, 0] }))
      : [{ shape: part.shape, size_mm: part.size_mm, material: part.material, offset: [0, 0, 0], rot: [0, 0, 0] }]
    const materials: THREE.MeshStandardMaterial[] = []
    const pieces: Piece[] = specs.map(c => {
      const sh = partShape({ shape: c.shape, size_mm: c.size_mm as Vec3 }, s)
      const m = surface(c.material as Material)
      materials.push(m)
      return { geometry: sh.geometry, scale: sh.scale, axis: sh.axis, position: c.offset.map(v => v * s) as Vec3,
        rotation: c.rot.map(d => d * DEG) as Vec3, material: m }
    })
    return { photo: false, pieces, materials }
  }, [part, s, anatomy, ground, texture])
}

/**
 * Explode amount eased toward the slider. During an assembly it holds until `delay` after Assemble was pressed,
 * then snaps home quickly; returns the eased value and when this item arrived (null outside an assembly).
 */
function useStaggeredExplode(explode: number, assembleAt: number | null, delay: number) {
  const eased = useRef(explode)
  return (dt: number, now: number) => {
    const arrive = assembleAt == null ? null : assembleAt + delay
    const snapping = arrive != null && now >= arrive && now < arrive + 600
    if (arrive == null || now >= arrive) eased.current += (explode - eased.current) * Math.min(1, dt * (snapping ? 16 : 8))
    return { value: eased.current, arrive }
  }
}

function PartMesh({ part, s, maxLayer, registry, explode, playback, focusIds, selectedId, onSelect, anatomy, ground, imageUrl, assembleAt, delay }:
  ViewerProps & { part: Part; s: number; maxLayer: number; registry: Registry; delay: number }) {
  const mover = useRef<THREE.Group>(null)
  const spinner = useRef<THREE.Group>(null)
  const [hovered, setHovered] = useState(false)
  const texture = useTexture(imageUrl)
  texture.colorSpace = THREE.SRGBColorSpace
  const { photo, pieces, materials } = usePieces(part, s, anatomy, ground, texture)
  const home = useMemo(() => new THREE.Vector3(...part.center_mm).multiplyScalar(s), [part.center_mm, s])
  const dir = useMemo(() => explodeDirection(part, s, maxLayer, new THREE.Vector3(...bodyBox(anatomy).center).multiplyScalar(s)),
    [part, s, maxLayer, anatomy])
  const tmp = useMemo(() => ({ pos: new THREE.Vector3(), glow: new THREE.Color() }), [])
  const step = useStaggeredExplode(explode, assembleAt, delay)
  const resting = useMemo(() => materials.map(m => m.opacity), [materials])

  useEffect(() => {
    const g = mover.current
    if (g) registry.set(part.id, g)
    return () => { registry.delete(part.id) }
  }, [part.id, registry])

  useFrame((_, dt) => {
    const g = mover.current, sp = spinner.current
    if (!g || !sp) return
    const now = performance.now()
    // Ease toward the slider so explode changes glide instead of jumping.
    const { value, arrive } = step(dt, now)
    tmp.pos.copy(home).addScaledVector(dir, value * 1.1)
    let glow = 0, opacity = 1
    tmp.glow.set(ACCENT)
    // Each part flashes as it locks into place during an assembly.
    if (arrive != null && now >= arrive && now < arrive + ARRIVAL_FLASH_MS) glow = 0.9 * (1 - (now - arrive) / ARRIVAL_FLASH_MS)
    sp.rotation.set(0, 0, 0)

    if (playback.kind === 'scenario') {
      const { index, progress, done } = scenarioCursor(playback.scenario, now - playback.startedAt)
      const steps = playback.scenario.steps
      const mine = steps.findIndex((st, i) => st.part_id === part.id && i === index)
      if (mine >= 0 && !done) {
        const st = steps[mine], w = Math.sin(progress * Math.PI)
        switch (st.effect) {
          case 'press': tmp.pos.addScaledVector(dir.clone().normalize(), -0.06 * w); glow = 0.8; break
          case 'rotate': sp.rotation.z = progress * Math.PI * 2; glow = 0.6; break
          case 'slide': tmp.pos.x += 0.07 * w; glow = 0.6; break
          case 'vibrate': tmp.pos.x += Math.sin(now * 0.09) * 0.012; tmp.pos.y += Math.cos(now * 0.11) * 0.012; glow = 0.7; break
          case 'heat': tmp.glow.set(EFFECT_GLOW.heat); glow = 0.4 + 0.6 * w; break
          case 'light': tmp.glow.set(EFFECT_GLOW.light); glow = 1.2 * w + 0.3; break
          case 'flow': tmp.glow.set(EFFECT_GLOW.flow); glow = 0.5 + 0.5 * w; break
          default: glow = 0.6 + 0.6 * w
        }
      } else if (steps.slice(0, index + 1).some(st => st.part_id === part.id)) {
        glow = 0.22
      } else {
        opacity = 0.35
      }
    } else if (playback.kind === 'failure') {
      const { access_path, culprit_part_id } = playback.failure
      const elapsed = now - playback.startedAt
      const k = access_path.indexOf(part.id)
      if (part.id === culprit_part_id) {
        tmp.glow.set(CULPRIT)
        const revealed = elapsed > access_path.length * REMOVE_MS
        glow = revealed ? 0.7 + 0.5 * Math.sin(now * 0.008) : 0.35
      } else if (k >= 0) {
        // Parts in the way lift off one at a time, outermost first.
        const q = clamp01((elapsed - k * REMOVE_MS) / REMOVE_MS)
        const e = 1 - Math.pow(1 - q, 3)
        tmp.pos.addScaledVector(dir.clone().normalize(), e * 1.6)
        opacity = 1 - 0.85 * e
      } else {
        // Bystanders recede once the culprit is revealed so nothing blocks the close-up.
        opacity = elapsed > access_path.length * REMOVE_MS ? 0.12 : 0.3
      }
    } else if (focusIds.length) {
      if (focusIds.includes(part.id)) glow = 0.55 + 0.25 * Math.sin(now * 0.004)
      else opacity = 0.45
    }
    // Accessories are context: they fade as they slide aside so the internals hold the eye.
    if (part.kind === 'accessory') opacity *= 1 - 0.7 * clamp01(value)
    if (part.id === selectedId) { glow = Math.max(glow, 0.8); tmp.glow.set(playback.kind === 'failure' && part.id === playback.failure.culprit_part_id ? CULPRIT : ACCENT) }
    if (hovered) glow = Math.max(glow, 0.35)

    g.position.lerp(tmp.pos, Math.min(1, dt * 10))
    materials.forEach((m, i) => {
      const target = opacity * resting[i]
      m.emissive.copy(tmp.glow)
      m.emissiveIntensity += (glow - m.emissiveIntensity) * Math.min(1, dt * 12)
      m.opacity += (target - m.opacity) * Math.min(1, dt * 10)
      m.transparent = m.opacity < 0.99
      // Keep writing depth while a part is still clearly visible, so its own walls don't show through each other.
      m.depthWrite = m.opacity > 0.2
    })
  })

  const label = labelFor(part, playback, selectedId, hovered)
  return (
    <group ref={mover} position={home}
      onClick={(e: ThreeEvent<MouseEvent>) => { e.stopPropagation(); onSelect(part.id) }}
      onPointerOver={(e: ThreeEvent<PointerEvent>) => { e.stopPropagation(); setHovered(true); document.body.style.cursor = 'pointer' }}
      onPointerOut={() => { setHovered(false); document.body.style.cursor = '' }}>
      {/* Photo pieces are already in the photo's front-view orientation, so the model's rotation does not apply. */}
      <group rotation={photo ? [0, 0, 0] : part.rotation_deg.map(d => d * DEG) as [number, number, number]}>
        <group ref={spinner}>
          {pieces.map((pc, i) => (
            <group key={i} position={pc.position} rotation={pc.rotation}>
              <group rotation={pc.axis}>
                <mesh geometry={pc.geometry} scale={pc.scale} material={pc.material} />
              </group>
            </group>
          ))}
        </group>
      </group>
      {label && (
        <Html position={[0, Math.max(...part.size_mm) * s * 0.6 + 0.06, 0]} zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
          <div className={`-translate-x-1/2 -translate-y-full whitespace-nowrap rounded-sm px-1 py-px font-mono text-caption shadow-ring ${label.tone === 'culprit' ? 'bg-red-text text-white' : 'bg-card text-ink'}`}>
            {label.text}
          </div>
        </Html>
      )}
    </group>
  )
}

function labelFor(part: Part, playback: Playback, selectedId: string | null, hovered: boolean): { text: string; tone: 'plain' | 'culprit' } | null {
  if (playback.kind === 'failure' && part.id === playback.failure.culprit_part_id) return { text: `Likely culprit · ${part.name}`, tone: 'culprit' }
  // Scenario step labels are drawn by ScenarioPath, which re-renders as steps advance.
  if (part.id === selectedId || hovered) return { text: part.name, tone: 'plain' }
  return null
}

/** Shell opacity: solid when assembled, thinning as the halves part, hidden while a scenario or failure plays. */
function shellOpacity(explode: number, xray: boolean, playback: Playback) {
  if (playback.kind !== 'idle') return 0
  if (xray) return 0.14
  return 1 - clamp01((explode - 0.2) / 0.5) * 0.65
}

function useShellFade(materials: THREE.Material[], target: () => number) {
  useFrame((_, dt) => {
    for (const m of materials) {
      m.opacity += (target() - m.opacity) * Math.min(1, dt * 8)
      m.transparent = m.opacity < 0.99
      m.depthWrite = m.opacity > 0.3
      // Skip drawing a faded-out shell entirely: its overlapping walls otherwise streak across the internals.
      m.visible = m.opacity > 0.02
    }
  })
}

/**
 * The photographed object as a 3D shell: SAM 3.1's outline extruded to Wonder's depth estimate with the photo on
 * the front. It is cut into a front and a back half that pull apart along the depth axis as the object explodes,
 * like a real teardown.
 */
function PhotoShell({ anatomy, ground, imageUrl, explode, xray, playback, assembleAt, delay }:
  { anatomy: Anatomy; ground: Ground; imageUrl: string; explode: number; xray: boolean; playback: Playback; assembleAt: number | null; delay: number }) {
  const texture = useTexture(imageUrl)
  texture.colorSpace = THREE.SRGBColorSpace
  const s = worldScale(anatomy)
  const depth = anatomy.object.dimensions_mm[2] * s
  const halves = useMemo(() => {
    if (!ground.object) return null
    const shapes = maskShapes(ground.object, anatomy, ground)
    const front = extrudeMask(shapes, anatomy, ground, 0, depth / 2, [FACE.cut, FACE.photo])
    const back = extrudeMask(shapes, anatomy, ground, -depth / 2, 0, [FACE.back, FACE.cut])
    return front && back ? { front, back } : null
  }, [anatomy, ground, depth])
  const materials = useMemo(() => {
    const tone = ground.object && texture.image ? maskColor(texture.image as HTMLImageElement, ground.object) : new THREE.Color('#3a3a3c')
    return [
      new THREE.MeshStandardMaterial({ map: texture, roughness: 0.5, transparent: true }),
      new THREE.MeshStandardMaterial({ color: tone, roughness: 0.55, transparent: true }),
      new THREE.MeshStandardMaterial({ color: tone.clone().multiplyScalar(0.45), roughness: 0.85, transparent: true, side: THREE.DoubleSide }),
      new THREE.MeshStandardMaterial({ color: tone, roughness: 0.55, transparent: true }),
    ]
  }, [texture, ground.object])
  const front = useRef<THREE.Mesh>(null), back = useRef<THREE.Mesh>(null)
  const step = useStaggeredExplode(explode, assembleAt, delay)
  const current = useRef(explode)
  // Opacity follows the halves' actual position, so the shell stays open until it closes in the assembly.
  useShellFade(materials, () => shellOpacity(current.current, xray, playback))
  useFrame((_, dt) => {
    current.current = step(dt, performance.now()).value
    const gap = current.current * 1.8
    if (front.current) front.current.position.z = gap
    if (back.current) back.current.position.z = -gap
  })
  if (!halves) return null
  return (
    <group>
      <mesh ref={front} geometry={halves.front} material={materials} renderOrder={10} raycast={() => null} />
      <mesh ref={back} geometry={halves.back} material={materials} renderOrder={10} raycast={() => null} />
    </group>
  )
}

function MeshShell({ url, anatomy, explode, xray, playback, assembleAt, delay }:
  { url: string; anatomy: Anatomy; explode: number; xray: boolean; playback: Playback; assembleAt: number | null; delay: number }) {
  const { scene } = useGLTF(url)
  const s = worldScale(anatomy)
  const { object, materials } = useMemo(() => {
    const root = scene.clone(true)
    // Fit SAM 3D's mesh to the object's estimated extent, centred on the origin.
    const box = new THREE.Box3().setFromObject(root)
    const size = box.getSize(new THREE.Vector3()), centre = box.getCenter(new THREE.Vector3())
    const [dw, dh, dd] = anatomy.object.dimensions_mm.map(v => v * s)
    root.position.sub(centre)
    const wrap = new THREE.Group()
    wrap.add(root)
    wrap.scale.set(dw / size.x, dh / size.y, dd / size.z)
    const mats: THREE.Material[] = []
    root.traverse(o => {
      const mesh = o as THREE.Mesh
      if (!mesh.isMesh) return
      mesh.raycast = () => null
      const m = (mesh.material as THREE.Material).clone()
      m.transparent = true
      mesh.material = m
      mats.push(m)
    })
    return { object: wrap, materials: mats }
  }, [scene, anatomy, s])
  const step = useStaggeredExplode(explode, assembleAt, delay)
  const current = useRef(explode)
  useFrame((_, dt) => { current.current = step(dt, performance.now()).value })
  useShellFade(materials, () => shellOpacity(current.current, xray, playback))
  return <primitive object={object} />
}

/** A signal pulse that travels from each step's part to the next, so the chain of cause reads at a glance. */
function ScenarioPath({ scenario, startedAt, registry }: { scenario: Scenario; startedAt: number; registry: Registry }) {
  const dot = useRef<THREE.Mesh>(null)
  const tag = useRef<THREE.Group>(null)
  const [points, setPoints] = useState<THREE.Vector3[]>([])
  const [step, setStep] = useState<{ index: number; done: boolean }>({ index: 0, done: false })
  const tmp = useMemo(() => ({ a: new THREE.Vector3(), b: new THREE.Vector3() }), [])

  useFrame(() => {
    const { index, progress, done } = scenarioCursor(scenario, performance.now() - startedAt)
    if (index !== step.index || done !== step.done) setStep({ index, done })
    const current = registry.get(scenario.steps[index]?.part_id)
    if (current && tag.current) current.getWorldPosition(tag.current.position)
    const pts = scenario.steps.slice(0, index + 1).map(st => registry.get(st.part_id)?.getWorldPosition(new THREE.Vector3()))
      .filter((v): v is THREE.Vector3 => !!v)
    if (pts.length !== points.length || pts.some((v, i) => !v.equals(points[i]))) setPoints(pts)
    const d = dot.current
    if (!d) return
    const from = index > 0 ? registry.get(scenario.steps[index - 1].part_id) : null
    const to = registry.get(scenario.steps[index].part_id)
    if (!from || !to || done || progress > 0.45) { d.visible = false; return }
    d.visible = true
    from.getWorldPosition(tmp.a); to.getWorldPosition(tmp.b)
    d.position.lerpVectors(tmp.a, tmp.b, clamp01(progress / 0.45))
  })

  return (
    <group>
      {points.length > 1 && <Line points={points} color={EFFECT_GLOW.flow} lineWidth={2} transparent opacity={0.7} dashed dashSize={0.04} gapSize={0.03} />}
      <mesh ref={dot} visible={false}>
        <sphereGeometry args={[0.03, 16, 16]} />
        <meshBasicMaterial color={EFFECT_GLOW.flow} toneMapped={false} />
      </mesh>
      <group ref={tag}>
        {!step.done && scenario.steps[step.index] && (
          <Html position={[0, 0.16, 0]} zIndexRange={[30, 0]} style={{ pointerEvents: 'none' }}>
            <div className="-translate-x-1/2 -translate-y-full whitespace-nowrap rounded-sm bg-accent-fill px-1 py-px font-mono text-caption text-white shadow-ring">
              <span className="tabular opacity-80">{String(step.index + 1).padStart(2, '0')}</span> {scenario.steps[step.index].label}
            </div>
          </Html>
        )}
      </group>
    </group>
  )
}
