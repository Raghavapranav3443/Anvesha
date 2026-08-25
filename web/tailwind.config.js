/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: 'var(--c-bg)',
        panel: 'var(--c-panel)',
        elev: 'var(--c-elev)',
        line: 'var(--c-line)',
        body: 'var(--c-text)',
        muted: 'var(--c-muted)',
        faint: 'var(--c-faint)',
        accent: { DEFAULT: 'var(--c-accent)', dim: 'var(--c-accent-dim)', soft: 'var(--c-accent-soft)' },
        good: 'var(--c-good)',
        warn: 'var(--c-warn)',
        bad: 'var(--c-bad)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
}
