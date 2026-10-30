export type Vec3 = [number, number, number]

export type Material =
  | 'plastic' | 'metal' | 'pcb' | 'battery' | 'rubber' | 'glass' | 'magnet' | 'fabric' | 'fluid' | 'organic' | 'ceramic' | 'other'
export type Shape = 'box' | 'rounded_box' | 'cylinder' | 'sphere' | 'capsule' | 'disc' | 'ring' | 'cone' | 'plate' | 'coil'
export type Effect = 'press' | 'rotate' | 'slide' | 'vibrate' | 'glow' | 'flow' | 'heat' | 'light'

export interface Part {
  id: string
  name: string
  kind: 'exterior' | 'internal'
  function: string
  material: Material
  shape: Shape
  size_mm: Vec3
  center_mm: Vec3
  rotation_deg: Vec3
  layer: number
  sam_prompt: string | null
  confidence: number
  grounded?: boolean
}

export interface Connection { from: string; to: string; kind: string; label: string }

export interface Step { part_id: string; effect: Effect; label: string; duration_ms: number }
export interface Scenario { id: string; title: string; summary: string; steps: Step[] }

export interface Failure {
  id: string
  symptom: string
  culprit_part_id: string
  mechanism: string
  likelihood: 'common' | 'occasional' | 'rare'
  checks: string[]
  fix: string
  access_path: string[]
}

export interface Anatomy {
  object: {
    name: string
    maker: string | null
    model: string | null
    category: string
    summary: string
    dimensions_mm: Vec3
    confidence: number
    evidence: string[]
    sam_prompt: string
  }
  parts: Part[]
  connections: Connection[]
  scenarios: Scenario[]
  failures: Failure[]
  caveats: string[]
}

export interface MaskMatch {
  box: [number, number, number, number]
  confidence: number | null
  mask_svg: string
  mask_bounds: [number, number, number, number]
  mask_size: [number, number]
}

export interface Ground {
  image_size: [number, number]
  object: MaskMatch | null
  parts: Record<string, MaskMatch>
  errors: Record<string, string>
}

export interface Answer {
  intent: 'explain' | 'scenario' | 'symptom' | 'locate' | 'disassemble' | 'other'
  text: string
  focus_part_ids: string[]
  scenario_id: string | null
  failure_id: string | null
  new_scenario: Scenario | null
  new_failure: Failure | null
}

export interface Session {
  id: string
  image_url: string
  size: [number, number]
  prompt: string
  history: string[]
  anatomy: Anatomy | null
  ground: Ground | null
  has_mesh: boolean
}

/** What the viewer is currently acting out. */
export type Playback =
  | { kind: 'idle' }
  | { kind: 'scenario'; scenario: Scenario; startedAt: number }
  | { kind: 'failure'; failure: Failure; startedAt: number }
