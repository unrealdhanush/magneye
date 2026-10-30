import { useEffect, useState } from 'react'

// Appearance preference. public/theme-init.js applies the stored choice before first paint; this keeps it in sync.
const THEMES = ['System', 'Light', 'Dark'] as const
type Theme = (typeof THEMES)[number]
const THEME_ICON: Record<Theme, string> = { System: '◐', Light: '☀', Dark: '☾' }

function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    try { const t = localStorage.getItem('magneye-theme'); return t === 'light' ? 'Light' : t === 'dark' ? 'Dark' : 'System' } catch { return 'System' }
  })
  useEffect(() => {
    const mq = matchMedia('(prefers-color-scheme: dark)')
    const apply = () => { document.documentElement.dataset.theme = theme === 'Dark' || (theme === 'System' && mq.matches) ? 'dark' : 'light' }
    apply()
    try { if (theme === 'System') localStorage.removeItem('magneye-theme'); else localStorage.setItem('magneye-theme', theme.toLowerCase()) } catch { /* storage unavailable */ }
    if (theme !== 'System') return
    mq.addEventListener('change', apply)
    return () => mq.removeEventListener('change', apply)
  }, [theme])
  return [theme, setTheme] as const
}

export function ThemeToggle() {
  const [theme, setTheme] = useTheme()
  return (
    <div role="group" aria-label="Appearance" className="inline-flex shrink-0 gap-1 rounded-md bg-ink/5 p-1">
      {THEMES.map(t => (
        <button key={t} type="button" aria-pressed={theme === t} aria-label={t} title={t} onClick={() => setTheme(t)}
          className={`grid h-6 w-6 place-items-center rounded-sm text-headline ${theme === t ? 'bg-card text-ink shadow' : 'text-ink hover:bg-card'}`}>
          <span aria-hidden="true">{THEME_ICON[t]}</span></button>
      ))}
    </div>
  )
}
