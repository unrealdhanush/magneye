import { Bounds, Html, Line, OrbitControls, useGLTF, useTexture } from '@react-three/drei'
import { Canvas, useFrame, type ThreeEvent } from '@react-three/fiber'
import { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import type { Anatomy, Ground, Part, Playback, Scenario } from '../types'
import { DEG, explodeDirection, partShape, silhouetteShell, worldScale } from './geometry'
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
}

/** Registry of each part's moving group, so pulses and labels can find parts in world space. */
type Registry = Map<string, THREE.Object3D>

export function Viewer(props: ViewerProps) {
  return (
    <Canvas camera={{ position: [3.4, 2.1, 5], fov: 38 }} dpr={[1, 2]} gl={{ antialias: true, alpha: true }}
      onPointerMissed={() => props.onSelect(null)}>
      <hemisphereLight args={['#ffffff', '#445566', 1.1]} />
      <directionalLight position={[3, 5, 4]} intensity={1.6} />
      <directionalLight position={[-4, -2, -3]} intensity={0.5} color="#88aaff" />
      <Suspense fallback={null}>
        {/* Frame the assembled object once; explode/playback then move parts within that view. */}
        <Bounds fit clip margin={1.9}>
          <Scene {...props} />
        </Bounds>
      </Suspense>
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} minDistance={1.2} maxDistance={12} />
    </Canvas>
  )
}

function Scene(p: ViewerProps) {
  const s = worldScale(p.anatomy)
  const registry = useMemo<Registry>(() => new Map(), [])
  const maxLayer = Math.max(0, ...p.anatomy.parts.map(x => x.layer))
  const ordered = useMemo(() => [...p.anatomy.parts].sort((a, b) => a.layer - b.layer), [p.anatomy.parts])

  return (
    <group>
      {ordered.map(part => (
        <PartMesh key={part.id} part={part} s={s} maxLayer={maxLayer} registry={registry} {...p} />
      ))}
      {p.ground && (p.meshUrl
        ? <MeshShell url={p.meshUrl} anatomy={p.anatomy} explode={p.explode} xray={p.xray} playback={p.playback} />
        : <PhotoShell anatomy={p.anatomy} ground={p.ground} imageUrl={p.imageUrl} explode={p.explode} xray={p.xray} playback={p.playback} />)}
      {p.playback.kind === 'scenario' && <ScenarioPath scenario={p.playback.scenario} startedAt={p.playback.startedAt} registry={registry} />}
    </group>
  )
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

function PartMesh({ part, s, maxLayer, registry, explode, playback, focusIds, selectedId, onSelect }:
  ViewerProps & { part: Part; s: number; maxLayer: number; registry: Registry }) {
  const mover = useRef<THREE.Group>(null)
  const spinner = useRef<THREE.Group>(null)
  const material = useRef<THREE.MeshStandardMaterial>(null)
  const [hovered, setHovered] = useState(false)
  const shape = useMemo(() => partShape(part, s), [part, s])
  const home = useMemo(() => new THREE.Vector3(...part.center_mm).multiplyScalar(s), [part.center_mm, s])
  const dir = useMemo(() => explodeDirection(part, s, maxLayer), [part, s, maxLayer])
  const look = MATERIAL_LOOK[part.material] ?? MATERIAL_LOOK.other
  const base = useMemo(() => new THREE.Color(look.color), [look.color])
  const tmp = useMemo(() => ({ pos: new THREE.Vector3(), glow: new THREE.Color() }), [])
  const eased = useRef(explode)

  useEffect(() => {
    const g = mover.current
    if (g) registry.set(part.id, g)
    return () => { registry.delete(part.id) }
  }, [part.id, registry])

  useFrame((_, dt) => {
    const g = mover.current, m = material.current, sp = spinner.current
    if (!g || !m || !sp) return
    const now = performance.now()
    // Ease toward the slider so explode changes glide instead of jumping.
    eased.current += (explode - eased.current) * Math.min(1, dt * 8)
    tmp.pos.copy(home).addScaledVector(dir, eased.current * 1.1)
    let glow = 0, opacity = 1
    tmp.glow.set(ACCENT)
    sp.rotation.set(0, 0, 0)

    if (playback.kind === 'scenario') {
      const { index, progress, done } = scenarioCursor(playback.scenario, now - playback.startedAt)
      const steps = playback.scenario.steps
      const mine = steps.findIndex((st, i) => st.part_id === part.id && i === index)
      if (mine >= 0 && !done) {
        const st = steps[mine], w = Math.sin(progress * Math.PI)
        switch (st.effect) {
          case 'press': tmp.pos.addScaledVector(dir.clone().normalize(), -0.06 * w); glow = 0.8; break
          case 'rotate': sp.rotation.y = progress * Math.PI * 2; glow = 0.6; break
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
        opacity = 0.3
      }
    } else if (focusIds.length) {
      if (focusIds.includes(part.id)) glow = 0.55 + 0.25 * Math.sin(now * 0.004)
      else opacity = 0.45
    }
    if (part.id === selectedId) { glow = Math.max(glow, 0.8); tmp.glow.set(playback.kind === 'failure' && part.id === playback.failure.culprit_part_id ? CULPRIT : ACCENT) }
    if (hovered) glow = Math.max(glow, 0.35)

    g.position.lerp(tmp.pos, Math.min(1, dt * 10))
    m.emissive.copy(tmp.glow)
    m.emissiveIntensity += (glow - m.emissiveIntensity) * Math.min(1, dt * 12)
    m.opacity += (opacity - m.opacity) * Math.min(1, dt * 10)
    m.transparent = m.opacity < 0.99
    m.depthWrite = m.opacity > 0.6
  })

  const label = labelFor(part, playback, selectedId, hovered)
  return (
    <group ref={mover} position={home}>
      <group rotation={part.rotation_deg.map(d => d * DEG) as [number, number, number]}>
        <group rotation={shape.axis}>
          <group ref={spinner}>
            <mesh geometry={shape.geometry} scale={shape.scale}
              onClick={(e: ThreeEvent<MouseEvent>) => { e.stopPropagation(); onSelect(part.id) }}
              onPointerOver={(e: ThreeEvent<PointerEvent>) => { e.stopPropagation(); setHovered(true); document.body.style.cursor = 'pointer' }}
              onPointerOut={() => { setHovered(false); document.body.style.cursor = '' }}>
              <meshStandardMaterial ref={material} color={base} metalness={look.metalness} roughness={look.roughness}
                emissive={ACCENT} emissiveIntensity={0} transparent opacity={1} />
            </mesh>
          </group>
        </group>
      </group>
      {label && (
        <Html position={[0, Math.max(...shape.scale) * 0.6 + 0.06, 0]} zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
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

/** Shell opacity: solid when assembled, fading as the object opens, faint while a scenario or failure plays. */
function shellOpacity(explode: number, xray: boolean, playback: Playback) {
  if (playback.kind !== 'idle') return 0.08
  if (xray) return 0.14
  return Math.max(0.06, 1 - explode * 2.2)
}

function useShellFade(materials: THREE.Material[], target: number) {
  useFrame((_, dt) => {
    for (const m of materials) {
      m.opacity += (target - m.opacity) * Math.min(1, dt * 8)
      m.transparent = m.opacity < 0.99
      m.depthWrite = m.opacity > 0.9
    }
  })
}

function PhotoShell({ anatomy, ground, imageUrl, explode, xray, playback }:
  { anatomy: Anatomy; ground: Ground; imageUrl: string; explode: number; xray: boolean; playback: Playback }) {
  const texture = useTexture(imageUrl)
  texture.colorSpace = THREE.SRGBColorSpace
  const geometry = useMemo(() => silhouetteShell(anatomy, ground), [anatomy, ground])
  const materials = useMemo(() => [
    new THREE.MeshStandardMaterial({ map: texture, roughness: 0.55, transparent: true, side: THREE.DoubleSide }),
    new THREE.MeshStandardMaterial({ color: '#9aa3b2', roughness: 0.5, metalness: 0.1, transparent: true }),
  ], [texture])
  useShellFade(materials, shellOpacity(explode, xray, playback))
  if (!geometry) return null
  return <mesh geometry={geometry} material={materials} renderOrder={10} raycast={() => null} />
}

function MeshShell({ url, anatomy, explode, xray, playback }:
  { url: string; anatomy: Anatomy; explode: number; xray: boolean; playback: Playback }) {
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
  useShellFade(materials, shellOpacity(explode, xray, playback))
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
