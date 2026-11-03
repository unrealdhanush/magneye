import { useEffect, useMemo, useRef, type RefObject } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { ContactShadows, RoundedBox } from '@react-three/drei'
import * as THREE from 'three'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'

/** Written by the hero's scroll loop, read every frame here. Plain numbers, so React never re-renders. */
export interface SceneState { e: number; px: number; py: number; phase: number }

interface Props {
  state: SceneState
  /** The numbered labels. Their screen position is set from here, so they follow the 3D parts. */
  labels: RefObject<HTMLElement | null>
  visible: boolean
  still: boolean
  onReady: () => void
}

// ---- Shape of the mouse -----------------------------------------------------------------------------------------
// Front is -z. The plan is a superellipse that narrows toward the front, the profile is a low dome with a hump at the back.
const HALF_W = 2.45, HALF_L = 5.8, SQUARE = 2.7
const half = (z: number) => { const f = Math.min(1, Math.max(0, -z / HALF_L)); return HALF_W * (1 - 0.3 * f * f) }
const rho = (x: number, z: number) => Math.pow(Math.pow(Math.abs(x) / half(z), SQUARE) + Math.pow(Math.abs(z) / HALF_L, SQUARE), 1 / SQUARE)
const crown = (z: number) => { const t = (z + HALF_L) / (2 * HALF_L); return 0.95 + 0.85 * Math.exp(-(((t - 0.64) / 0.28) ** 2)) }
const surface = (x: number, z: number) => { const r = rho(x, z); return r >= 1 ? 0 : crown(z) * Math.pow(1 - r ** 3, 0.45) }
const sgnPow = (v: number) => Math.sign(v) * Math.pow(Math.abs(v), 2 / SQUARE)

function shellGeometry() {
  const R = 40, S = 112, pos: number[] = [], idx: number[] = []
  for (let i = 0; i <= R; i++) {
    const r = Math.sin((i / R) * Math.PI / 2) // rings bunch up at the rim, where the surface turns fastest
    for (let j = 0; j <= S; j++) {
      const th = (j / S) * Math.PI * 2
      const z = r * HALF_L * sgnPow(Math.sin(th))
      const x = r * half(z) * sgnPow(Math.cos(th))
      pos.push(x, crown(z) * Math.pow(1 - r ** 3, 0.45), z)
    }
  }
  for (let i = 0; i < R; i++) for (let j = 0; j < S; j++) {
    const a = i * (S + 1) + j, b = a + S + 1
    idx.push(a, b, a + 1, a + 1, b, b + 1)
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.setIndex(idx)
  g.computeVertexNormals()
  // Close the seam and the pole so shading is continuous.
  const n = g.getAttribute('normal') as THREE.BufferAttribute
  const v = new THREE.Vector3(), w = new THREE.Vector3()
  for (let i = 1; i <= R; i++) {
    const a = i * (S + 1), b = a + S
    v.fromBufferAttribute(n, a).add(w.fromBufferAttribute(n, b)).normalize()
    n.setXYZ(a, v.x, v.y, v.z); n.setXYZ(b, v.x, v.y, v.z)
  }
  for (let j = 0; j <= S; j++) n.setXYZ(j, 0, 1, 0)
  return g
}

function outline(scale: number, steps = 96) {
  const pts: THREE.Vector2[] = []
  for (let j = 0; j < steps; j++) {
    const th = (j / steps) * Math.PI * 2
    const z = scale * HALF_L * sgnPow(Math.sin(th))
    pts.push(new THREE.Vector2(scale * half(z) * sgnPow(Math.cos(th)), -z))
  }
  return pts
}

function baseGeometry() {
  const g = new THREE.ExtrudeGeometry(new THREE.Shape(outline(0.97)), { depth: 0.22, bevelEnabled: true, bevelSize: 0.06, bevelThickness: 0.05, bevelSegments: 2, curveSegments: 1 })
  g.rotateX(-Math.PI / 2)
  g.translate(0, -0.22, 0)
  return g
}

/** A scroll wheel with square notches round the rim, axis along x. */
function wheelGeometry() {
  const teeth = 24, r0 = 0.55, r1 = 0.47, pts: THREE.Vector2[] = []
  for (let k = 0; k < teeth; k++) {
    for (const [f, r] of [[0, r0], [0.5, r0], [0.5, r1], [1, r1]] as const) {
      const a = ((k + f) / teeth) * Math.PI * 2
      pts.push(new THREE.Vector2(Math.cos(a) * r, Math.sin(a) * r))
    }
  }
  const g = new THREE.ExtrudeGeometry(new THREE.Shape(pts), { depth: 0.4, bevelEnabled: false })
  g.translate(0, 0, -0.2)
  g.rotateY(Math.PI / 2)
  return g
}

/** Circuit-board artwork: traces, pads and vias on dark teal. Also used as the emissive map, so the copper glows. */
function boardTexture() {
  const c = document.createElement('canvas'); c.width = 512; c.height = 1024
  const g = c.getContext('2d')!
  g.fillStyle = '#082623'; g.fillRect(0, 0, 512, 1024)
  let seed = 7
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647
  g.lineCap = 'round'; g.lineJoin = 'round'
  for (let k = 0; k < 70; k++) {
    let x = Math.floor(rnd() * 16) * 32 + 16, y = Math.floor(rnd() * 32) * 32 + 16
    g.strokeStyle = rnd() > 0.25 ? '#34e8c4' : '#1b8f7c'; g.lineWidth = rnd() > 0.6 ? 5 : 3
    g.beginPath(); g.moveTo(x, y)
    for (let s = 0; s < 4; s++) {
      if (rnd() > 0.5) x += (rnd() > 0.5 ? 1 : -1) * (1 + Math.floor(rnd() * 5)) * 32
      else y += (rnd() > 0.5 ? 1 : -1) * (1 + Math.floor(rnd() * 8)) * 32
      x = Math.min(496, Math.max(16, x)); y = Math.min(1008, Math.max(16, y))
      g.lineTo(x, y)
    }
    g.stroke()
    g.fillStyle = '#9ffbe6'; g.beginPath(); g.arc(x, y, 7, 0, Math.PI * 2); g.fill()
    g.fillStyle = '#082623'; g.beginPath(); g.arc(x, y, 3, 0, Math.PI * 2); g.fill()
  }
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8
  return t
}

const lerp = (a: number, b: number, k: number) => a + (b - a) * k
const std = (color: string, extra: Partial<THREE.MeshStandardMaterialParameters> = {}) =>
  new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.1, transparent: true, ...extra })

// Where each part rests, and how far it lifts at full explosion. Same order and names as the labels beside the scene.
const REST = { base: 0, battery: 0.6, board: 0.55, sw: 0.58, wheel: 1.12, shell: 0.33 }
const LIFT = { base: -1.2, battery: 0.8, board: 2.6, sw: 4.6, wheel: 6.8, shell: 9.6 }
const ANCHORS: Record<string, { x: number; z: number; rest: number; lift: number }> = {
  shell: { x: -2.8, z: 1, rest: REST.shell + 0.9, lift: LIFT.shell },
  wheel: { x: -1.6, z: -2.4, rest: REST.wheel, lift: LIFT.wheel },
  switch: { x: -2.0, z: -3.3, rest: REST.sw, lift: LIFT.sw },
  board: { x: -2.3, z: 0.5, rest: REST.board, lift: LIFT.board },
  battery: { x: -1.6, z: 2.5, rest: REST.battery, lift: LIFT.battery },
}

const readDark = () => {
  const bg = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim().split(/\s+/).map(Number)
  return (bg[0] ?? 0) < 128
}

function Studio() {
  const { gl, scene, invalidate } = useThree()
  useEffect(() => {
    const pm = new THREE.PMREMGenerator(gl)
    const env = pm.fromScene(new RoomEnvironment(), 0.04).texture
    scene.environment = env
    invalidate()
    return () => { scene.environment = null; env.dispose(); pm.dispose() }
  }, [gl, scene, invalidate])
  return null
}

function Mouse({ state, labels, still, onReady }: Omit<Props, 'visible'>) {
  const { invalidate } = useThree()
  const rig = useRef<THREE.Group>(null)
  const g = { base: useRef<THREE.Group>(null), battery: useRef<THREE.Group>(null), board: useRef<THREE.Group>(null), sw: useRef<THREE.Group>(null), wheel: useRef<THREE.Mesh>(null), shell: useRef<THREE.Group>(null) }
  const dark = useRef(true)
  const shine = useRef<THREE.PointLight>(null)

  const geo = useMemo(() => ({ shell: shellGeometry(), base: baseGeometry(), wheel: wheelGeometry() }), [])
  const tex = useMemo(() => boardTexture(), [])
  const m = useMemo(() => ({
    shell: new THREE.MeshPhysicalMaterial({ color: '#5b6577', roughness: 0.3, metalness: 0.05, clearcoat: 1, clearcoatRoughness: 0.12, transparent: true, side: THREE.DoubleSide }),
    groove: new THREE.MeshBasicMaterial({ color: '#0b0c0f', transparent: true }),
    base: std('#454b57', { roughness: 0.6 }),
    battery: std('#3a5775', { roughness: 0.4, metalness: 0.2 }),
    batteryLabel: std('#d3dce6', { roughness: 0.7, metalness: 0 }),
    board: std('#ffffff', { map: tex, emissiveMap: tex, emissive: new THREE.Color('#7ffbe0'), emissiveIntensity: 0.45, roughness: 0.45 }),
    chip: std('#101216', { roughness: 0.35, metalness: 0.6 }),
    cap: std('#c8a24a', { roughness: 0.3, metalness: 0.8 }),
    wheel: std('#2a2e35', { roughness: 0.55, metalness: 0.3 }),
    glow: new THREE.MeshStandardMaterial({ color: '#0b3a44', emissive: new THREE.Color('#38d8ff'), emissiveIntensity: 1.8, transparent: true }),
    swBody: std('#17191e', { roughness: 0.5 }),
    swTop: std('#d9dae0', { roughness: 0.35, metalness: 0.4 }),
    swBodyL: std('#17191e', { roughness: 0.5 }),
    swTopL: std('#d9dae0', { roughness: 0.35, metalness: 0.4 }),
  }), [tex])

  const grooves = useMemo(() => {
    const line = (pts: [number, number][], r: number) => {
      const c = new THREE.CatmullRomCurve3(pts.map(([x, z]) => new THREE.Vector3(x, surface(x, z) + 0.012, z)))
      return new THREE.TubeGeometry(c, 48, r, 6, false)
    }
    const cross: [number, number][] = []
    for (let k = 0; k <= 12; k++) { const x = -2.0 + (k / 12) * 4.0; cross.push([x, -1.1 + 0.25 * Math.pow(x / 2.0, 2)]) }
    return [line([[0, -5.3], [0, -3.8], [0, -2.6], [0, -1.15]], 0.04), line(cross, 0.035)]
  }, [])
  useEffect(() => () => { for (const x of [...Object.values(geo), ...grooves, tex, ...Object.values(m)]) x.dispose() }, [geo, grooves, tex, m])

  // Colours that follow the page theme.
  useEffect(() => {
    const read = () => { dark.current = readDark(); invalidate() }
    read()
    const mo = new MutationObserver(read)
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class', 'style'] })
    const mq = matchMedia('(prefers-color-scheme: dark)')
    mq.addEventListener('change', read)
    return () => { mo.disconnect(); mq.removeEventListener('change', read) }
  }, [invalidate])

  const els = useRef<HTMLElement[]>([])
  useEffect(() => { els.current = Array.from(labels.current?.querySelectorAll<HTMLElement>('[data-id]') ?? []) }, [labels])

  const cur = useRef({ dim: 0, e: 0, ry: 0, rx: 0, shell: new THREE.Color('#5b6577'), frames: 0 })
  const v = useMemo(() => new THREE.Vector3(), [])
  const red = useMemo(() => new THREE.Color('#ff3b30'), [])
  const plain = useMemo(() => ({ body: new THREE.Color('#17191e'), top: new THREE.Color('#d9dae0') }), [])
  const pearl = useMemo(() => new THREE.Color('#5b6577'), [])
  const graphite = useMemo(() => new THREE.Color('#8e99ad'), [])

  useFrame(({ camera, size, clock, scene }, dt) => {
    scene.environmentIntensity = dark.current ? 0.65 : 0.7
    const s = state, c = cur.current, t = still ? 0 : clock.elapsedTime
    const k = still ? 1 : 1 - Math.pow(0.001, dt) // frame-rate independent easing
    c.e = lerp(c.e, s.e, k)
    const dimT = s.phase === 2 ? 1 : 0
    c.dim = lerp(c.dim, dimT, still ? 1 : 1 - Math.pow(0.02, dt))
    const e = c.e

    const R = rig.current
    if (R) {
      c.ry = lerp(c.ry, s.px * 0.22, k); c.rx = lerp(c.rx, s.py * 0.05, k)
      R.rotation.set(c.rx, Math.PI - 0.6 + e * 0.35 + c.ry, 0)
      R.position.y = (1 - e) * Math.sin(t * 0.9) * 0.12
    }
    const at = (r: { current: THREE.Object3D | null }, rest: number, lift: number) => { if (r.current) r.current.position.y = rest + lift * e }
    at(g.base, REST.base, LIFT.base); at(g.battery, REST.battery, LIFT.battery); at(g.board, REST.board, LIFT.board)
    at(g.sw, REST.sw, LIFT.sw); at(g.wheel, REST.wheel, LIFT.wheel); at(g.shell, REST.shell, LIFT.shell)
    if (g.shell.current) g.shell.current.rotation.x = -0.12 * e
    if (g.wheel.current) g.wheel.current.rotation.x = e * Math.PI * 2 + t * 0.15

    // Fault phase: everything but the left switch fades back, the shell nearly vanishes, the switch turns red and pulses.
    const d = c.dim, pulse = still ? 0.6 : 0.5 + 0.5 * Math.sin(t * 4.2)
    const ghost = 1 - d * 0.66
    for (const x of [m.base, m.battery, m.batteryLabel, m.board, m.chip, m.cap, m.wheel, m.glow, m.swBody, m.swTop]) x.opacity = ghost
    m.groove.opacity = 1 - d * 0.93
    m.shell.opacity = 1 - d * 0.935
    m.shell.depthWrite = m.shell.opacity > 0.98
    m.board.emissiveIntensity = 0.45 * (1 - d * 0.7)
    m.glow.emissiveIntensity = 1.8 * (1 - d * 0.7)
    m.swBodyL.color.copy(plain.body).lerp(red, d * 0.55)
    m.swTopL.color.copy(plain.top).lerp(red, d)
    m.swTopL.emissive.copy(red); m.swTopL.emissiveIntensity = d * (0.5 + pulse * 1.3)
    m.swBodyL.emissive.copy(red); m.swBodyL.emissiveIntensity = d * (0.15 + pulse * 0.5)
    if (shine.current) shine.current.intensity = d * (20 + pulse * 60)

    c.shell.lerp(dark.current ? pearl : graphite, still ? 1 : 0.12)
    m.shell.color.copy(c.shell)

    // Camera pulls up to follow the stack as it rises.
    const narrow = size.width < 640
    camera.position.set(narrow ? 2.4 : 0, 12 + e * 2.5, 31)
    camera.lookAt(narrow ? 2.4 : 0, 1.2 + e * 3.8, 0)
    camera.updateMatrixWorld()

    // Labels ride along with their parts. Where two would collide, the lower one is nudged down.
    R?.updateWorldMatrix(true, false)
    const rows: { el: HTMLElement; x: number; y: number }[] = []
    for (const el of els.current) {
      const a = ANCHORS[el.dataset.id ?? '']
      if (!a || !R) continue
      v.set(a.x, a.rest + a.lift * e, a.z).applyMatrix4(R.matrixWorld).project(camera)
      rows.push({ el, x: (v.x * 0.5 + 0.5) * size.width, y: (-v.y * 0.5 + 0.5) * size.height })
    }
    rows.sort((p, q) => p.y - q.y)
    const gap = size.width < 640 ? 28 : 40
    rows.forEach((r, i) => {
      const prev = rows[i - 1]
      if (prev && r.y < prev.y + gap) r.y = prev.y + gap
      r.el.style.setProperty('--lx', `${r.x.toFixed(1)}px`)
      r.el.style.setProperty('--ly', `${r.y.toFixed(1)}px`)
    })

    if (++c.frames === 3) onReady()
  })

  return (
    <>
      <ambientLight intensity={0.15} />
      <directionalLight position={[8, 16, 10]} intensity={1.5} />
      <directionalLight position={[-12, 6, -8]} intensity={1.4} color="#5aa8ff" />
      <pointLight ref={shine} position={[-1.4, 8, -3.5]} color="#ff3b30" intensity={0} distance={14} />

      <group ref={rig}>
        <group ref={g.base}>
          <mesh geometry={geo.base} material={m.base} />
          <mesh position={[0, -0.12, 3.6]} rotation={[-Math.PI / 2, 0, 0]}><circleGeometry args={[0.55, 24]} /><meshBasicMaterial color="#05060a" /></mesh>
        </group>

        <group ref={g.battery}>
          <RoundedBox args={[2.0, 0.5, 3.6]} radius={0.14} smoothness={3} position={[0, 0, 2.5]} material={m.battery} />
          <mesh position={[0, 0.27, 2.5]} material={m.batteryLabel}><boxGeometry args={[1.5, 0.03, 2.7]} /></mesh>
        </group>

        <group ref={g.board}>
          <RoundedBox args={[3.0, 0.12, 8.0]} radius={0.05} smoothness={2} position={[0, 0, 0.1]} material={m.board} />
          <mesh position={[0, 0.12, -0.6]} material={m.chip}><boxGeometry args={[1.3, 0.14, 1.3]} /></mesh>
          <mesh position={[-0.7, 0.1, 1.5]} material={m.chip}><boxGeometry args={[0.9, 0.1, 0.9]} /></mesh>
          <mesh position={[0.7, 0.1, 2.4]} material={m.chip}><boxGeometry args={[0.7, 0.1, 1.4]} /></mesh>
          {[[-1.1, -1.4], [-0.8, -1.4], [-0.5, -1.4], [1.0, 0.6], [1.0, 1.0]].map(([x, z], i) => (
            <mesh key={i} position={[x, 0.13, z]} material={m.cap}><cylinderGeometry args={[0.13, 0.13, 0.2, 12]} /></mesh>
          ))}
        </group>

        <group ref={g.sw}>
          {[-1, 1].map(side => {
            const left = side < 0
            return (
              <group key={side} position={[side * 0.8, 0, -3.3]}>
                <RoundedBox args={[1.1, 0.5, 1.1]} radius={0.08} smoothness={2} material={left ? m.swBodyL : m.swBody} />
                <RoundedBox args={[0.7, 0.22, 0.5]} radius={0.06} smoothness={2} position={[0, 0.31, 0]} material={left ? m.swTopL : m.swTop} />
              </group>
            )
          })}
        </group>

        <mesh ref={g.wheel} geometry={geo.wheel} material={m.wheel} position={[0, REST.wheel, -2.4]}>
          {[-0.21, 0.21].map(x => (
            <mesh key={x} position={[x, 0, 0]} rotation={[0, Math.PI / 2, 0]} material={m.glow}><circleGeometry args={[0.34, 24]} /></mesh>
          ))}
        </mesh>

        <group ref={g.shell}>
          <mesh geometry={geo.shell} material={m.shell} />
          {grooves.map((gr, i) => <mesh key={i} geometry={gr} material={m.groove} />)}
          <mesh position={[0, surface(0, -2.4) + 0.02, -2.4]} material={m.groove}><boxGeometry args={[0.75, 0.03, 1.3]} /></mesh>
        </group>
      </group>

      <ContactShadows position={[0, -1.75, 0]} opacity={0.5} scale={26} blur={2.6} far={7} resolution={512} />
      <mesh position={[0, -1.74, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[7.2, 7.27, 96]} />
        <meshBasicMaterial color="#38d8ff" transparent opacity={0.35} />
      </mesh>
    </>
  )
}

export default function MouseScene({ state, labels, visible, still, onReady }: Props) {
  return (
    <Canvas
      dpr={[1, 2]}
      frameloop={still ? 'demand' : visible ? 'always' : 'never'}
      camera={{ fov: 24, near: 0.1, far: 120, position: [0, 12, 31] }}
      gl={{ antialias: true, alpha: true, powerPreference: 'high-performance', toneMapping: THREE.NeutralToneMapping }}
      style={{ position: 'absolute', inset: 0 }}
      aria-hidden="true"
    >
      <Studio />
      <Mouse state={state} labels={labels} still={still} onReady={onReady} />
    </Canvas>
  )
}
