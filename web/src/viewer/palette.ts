import type { Material } from '../types'

// Colours for 3D materials (scene colours, not UI chrome). Same hue per material everywhere.
export const MATERIAL_LOOK: Record<Material, { color: string; metalness: number; roughness: number; label: string }> = {
  plastic: { color: '#c7c9d1', metalness: 0.0, roughness: 0.55, label: 'Plastic' },
  metal: { color: '#a8b0bd', metalness: 0.85, roughness: 0.3, label: 'Metal' },
  pcb: { color: '#1f8f5a', metalness: 0.1, roughness: 0.6, label: 'Circuit board' },
  battery: { color: '#ff9f0a', metalness: 0.3, roughness: 0.4, label: 'Battery' },
  rubber: { color: '#3a3a3c', metalness: 0.0, roughness: 0.9, label: 'Rubber' },
  glass: { color: '#64d2ff', metalness: 0.0, roughness: 0.05, label: 'Glass / lens' },
  magnet: { color: '#5e5ce6', metalness: 0.6, roughness: 0.35, label: 'Magnet' },
  fabric: { color: '#bf5af2', metalness: 0.0, roughness: 1.0, label: 'Fabric' },
  fluid: { color: '#0a84ff', metalness: 0.0, roughness: 0.1, label: 'Fluid' },
  organic: { color: '#ff6482', metalness: 0.0, roughness: 0.7, label: 'Organic' },
  ceramic: { color: '#f2f2f7', metalness: 0.0, roughness: 0.3, label: 'Ceramic' },
  other: { color: '#8e8e93', metalness: 0.2, roughness: 0.5, label: 'Other' },
}

export const ACCENT = '#0a84ff'
export const CULPRIT = '#ff453a'
export const EFFECT_GLOW = { glow: '#0a84ff', flow: '#64d2ff', heat: '#ff9f0a', light: '#ffd60a' } as const
