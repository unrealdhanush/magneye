// Design tokens. The scales below replace Tailwind's defaults, so only these values exist as classes.
const px = n => `${n}px`
const v = name => `rgb(var(--${name}) / <alpha-value>)`
const hue = name => ({ DEFAULT: v(name), text: v(`${name}-text`) })

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    colors: {
      transparent: 'transparent', current: 'currentColor', white: '#ffffff', black: '#000000',
      ink: v('ink'), muted: v('muted'), bg: v('bg'), card: v('card'),
      accent: { DEFAULT: v('accent'), text: v('accent-text'), fill: v('accent-fill'), strong: v('accent-strong') },
      blue: hue('blue'), indigo: hue('indigo'), purple: hue('purple'), pink: hue('pink'),
      orange: hue('orange'), teal: hue('teal'), green: hue('green'), red: hue('red'),
      success: hue('green'), destructive: hue('red'), warning: hue('orange'),
    },
    spacing: { 0: '0', px: '1px', 1: px(8), 1.5: px(12), 2: px(16), 2.5: px(20), 3: px(24), 4: px(32), 6: px(48), 8: px(64) },
    borderRadius: { none: '0', sm: px(8), md: px(12), lg: px(16), xl: px(24), full: '9999px' },
    boxShadow: {
      none: 'none', DEFAULT: '0 4px 8px var(--shadow)', glass: '0 8px 32px var(--shadow), inset 0 1px 0 var(--glass-inset)',
      ring: 'var(--shadow-border)', 'ring-hover': 'var(--shadow-border-hover)',
    },
    fontSize: {
      largeTitle: [px(34), { lineHeight: px(41), fontWeight: '700' }],
      title: [px(28), { lineHeight: px(34), fontWeight: '700' }],
      title2: [px(22), { lineHeight: px(28), fontWeight: '700' }],
      title3: [px(20), { lineHeight: px(25), fontWeight: '600' }],
      headline: [px(17), { lineHeight: px(22), fontWeight: '600' }],
      body: [px(17), px(22)],
      callout: [px(16), px(21)],
      subheadline: [px(15), px(20)],
      footnote: [px(13), px(18)],
      caption: [px(12), px(16)],
      caption2: [px(11), px(13)],
    },
    fontFamily: {
      sans: ['-apple-system', 'BlinkMacSystemFont', '"SF Pro Text"', '"Helvetica Neue"', 'Arial', 'sans-serif'],
      mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      display: ['Michroma', '-apple-system', 'sans-serif'],
    },
    extend: {
      borderColor: { DEFAULT: 'var(--line)' },
      ringColor: { DEFAULT: 'rgb(var(--accent))' },
      minHeight: { tap: px(48) },
      minWidth: { tap: px(48), table: px(640) },
      maxWidth: { page: px(1600) },
      // Very wide displays get a fourth column (object details beside the conversation).
      screens: { '3xl': px(2200) },
      transitionTimingFunction: { out: 'cubic-bezier(0.23, 1, 0.32, 1)', 'in-out': 'cubic-bezier(0.77, 0, 0.175, 1)' },
      scale: { press: '0.96' },
    },
  },
  plugins: [],
}
