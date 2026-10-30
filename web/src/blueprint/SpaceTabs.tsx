import type { KeyboardEvent } from 'react'

export type Space = 'reality' | 'blueprint'
const SPACES: { id: Space; label: string }[] = [{ id: 'reality', label: 'Reality' }, { id: 'blueprint', label: 'Blueprint' }]

/** The switch between the 3D Reality space and the Blueprint space; a pill slides under the active tab. */
export function SpaceTabs({ space, onChange }: { space: Space; onChange: (s: Space) => void }) {
  const index = SPACES.findIndex(s => s.id === space)
  const onKey = (e: KeyboardEvent) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
    e.preventDefault()
    onChange(SPACES[(index + (e.key === 'ArrowRight' ? 1 : SPACES.length - 1)) % SPACES.length].id)
  }
  return (
    <div role="tablist" aria-label="Space" onKeyDown={onKey}
      className="glass relative grid grid-cols-2 rounded-lg p-px">
      <span aria-hidden="true" className="absolute inset-y-px left-px w-[calc(50%-1px)] rounded-[7px] bg-accent-fill transition-transform duration-300 ease-out"
        style={{ transform: `translateX(${index * 100}%)` }} />
      {SPACES.map(s => (
        <button key={s.id} type="button" role="tab" aria-selected={space === s.id} tabIndex={space === s.id ? 0 : -1}
          aria-controls={`space-${s.id}`} id={`tab-${s.id}`} onClick={() => onChange(s.id)}
          className={`press relative min-h-tap px-2 text-subheadline font-medium transition-colors duration-200 ${space === s.id ? 'text-white' : 'text-ink hover:text-accent-text'}`}>
          {s.label}
        </button>
      ))}
    </div>
  )
}
