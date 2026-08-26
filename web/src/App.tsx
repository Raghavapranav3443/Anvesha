import { lazy, Suspense, useEffect, useState } from 'react'
import type { ComponentType } from 'react'
import Console from './components/Console'
import ProvenanceView from './components/Provenance'
import HistoryView from './components/HistoryView'
import EvaluationView from './components/EvaluationView'
import HelpView from './components/HelpView'
import Onboarding from './components/Onboarding'
import { ErrorBoundary } from './components/ErrorBoundary'
import { fetchProvenance, type Provenance } from './api'

// ---- removable landing page -------------------------------------------- #
// Discovered via glob: deleting web/src/landing/ removes the feature with
// zero code edits (the glob then resolves to {} at build time).
// See web/src/landing/README.md.
const landingModules = import.meta.glob('./landing/index.tsx')
const hasLanding = Object.keys(landingModules).length > 0
const LandingPage = hasLanding
  ? lazy(landingModules['./landing/index.tsx'] as () => Promise<{
      default: ComponentType<{ onEnterConsole?: () => void }>
    }>)
  : null

type View = 'home' | 'console' | 'history' | 'evaluation' | 'provenance' | 'help'

const NAV: { id: View; label: string; icon: string; hint: string }[] = [
  { id: 'console', label: 'Console', icon: '🛰️', hint: 'Upload imagery & ask questions' },
  { id: 'history', label: 'History', icon: '🗂️', hint: 'Past analyses & comparison' },
  { id: 'evaluation', label: 'Evaluation', icon: '📊', hint: 'Measured benchmark scorecard' },
  { id: 'provenance', label: 'Provenance', icon: '🧠', hint: 'Models, training data, metrics' },
  { id: 'help', label: 'Help', icon: '❓', hint: 'Glossary & how it works' },
]

function OrbitMark() {
  return (
    <div className="relative h-10 w-10 shrink-0">
      <svg viewBox="0 0 40 40" className="h-10 w-10">
        <circle cx="20" cy="20" r="8.5" className="fill-none stroke-accent" strokeWidth="1.6" />
        <ellipse cx="20" cy="20" rx="16" ry="6.5" fill="none" stroke="var(--c-line)"
          strokeWidth="1" transform="rotate(-18 20 20)" />
      </svg>
      <span className="orbit-dot absolute left-1/2 top-1/2 -ml-[3px] -mt-[3px] h-1.5 w-1.5 rounded-full bg-accent shadow-[0_0_6px_rgba(76,141,246,.9)]" />
    </div>
  )
}

export default function App() {
  const [view, setView] = useState<View>(hasLanding ? 'home' : 'console')
  const [prov, setProv] = useState<Provenance | null>(null)
  const [theme, setTheme] = useState<'light' | 'dark'>(
    () => (localStorage.getItem('anvesha-theme') === 'dark' ? 'dark' : 'light'))
  const [onboard, setOnboard] = useState(
    () => !localStorage.getItem('anvesha-onboarded'))

  useEffect(() => { fetchProvenance().then(setProv).catch(() => {}) }, [])

  function toggleTheme() {
    const next = theme === 'light' ? 'dark' : 'light'
    setTheme(next)
    document.documentElement.dataset.theme = next
    localStorage.setItem('anvesha-theme', next)
  }

  return (
    <div className="flex min-h-full overflow-x-hidden">
      {/* ---- left rail nav (hidden on the landing view) ---- */}
      {view !== 'home' && (
      <aside className="fixed inset-y-0 left-0 z-40 flex w-[76px] flex-col items-center border-r border-line bg-panel py-4">
        <div className="mb-6">
          {hasLanding ? (
            <button onClick={() => setView('home')} title="Mission overview"
              className="rounded-lg p-1 transition-colors hover:bg-elev">
              <OrbitMark />
            </button>
          ) : (
            <OrbitMark />
          )}
        </div>
        <nav className="flex flex-1 flex-col gap-1.5">
          {NAV.map(n => (
            <button key={n.id} onClick={() => setView(n.id)} title={n.hint}
              className={`group flex w-[60px] flex-col items-center gap-0.5 rounded-lg py-2 transition-colors ${
                view === n.id ? 'bg-accent-soft text-accent' : 'text-muted hover:bg-elev hover:text-body'}`}>
              <span className="text-[18px] leading-none">{n.icon}</span>
              <span className="text-[11px] font-medium leading-tight text-center">{n.label}</span>
            </button>
          ))}
        </nav>
        <button onClick={toggleTheme} title="Toggle light/dark theme"
          className="flex w-[60px] flex-col items-center gap-0.5 rounded-lg py-2 text-muted hover:bg-elev hover:text-body">
          <span className="text-[17px]">{theme === 'light' ? '🌙' : '☀️'}</span>
          <span className="text-[11px] font-medium leading-tight">{theme === 'light' ? 'Dark' : 'Light'}</span>
        </button>
      </aside>
      )}

      {/* ---- main column ---- */}
      <div className={view === 'home'
        ? 'min-w-0 flex-1'
        : 'ml-[76px] flex min-h-full min-w-0 flex-1 flex-col'}>
        {view !== 'home' && (
        <header className="sticky top-0 z-30 border-b border-line bg-panel/90 backdrop-blur">
          <div className="relative flex h-[68px] items-center gap-4 overflow-hidden px-7">
            {/* signature animation: satellite pass */}
            <span className="sat-pass pointer-events-none absolute inset-x-0 top-1/2 h-px bg-gradient-to-r from-transparent via-accent/40 to-transparent" />
            <div className="relative z-10 mr-auto">
              <div className="text-[19px] font-bold tracking-tight text-body">Anvesha</div>
              <div className="-mt-0.5 font-mono text-[14px] uppercase tracking-[.16em] text-faint">
                Earth Observation &amp; Investigation System · SIH26167
              </div>
            </div>
            <span className="hidden items-center gap-1.5 rounded-full border border-line px-3 py-1 text-[14px] text-muted md:flex">
              <span className="h-2 w-2 rounded-full bg-good" /> system ready
            </span>
          </div>
        </header>
        )}

        <main className={view === 'home'
          ? 'min-w-0 flex-1'
          : 'mx-auto w-full max-w-7xl flex-1 px-7 py-8'}>
          {view === 'home' && LandingPage && (
            <Suspense fallback={
              <div className="grid h-[80vh] place-items-center font-mono text-xs uppercase tracking-[.3em] text-faint">
                initialising orbit…
              </div>
            }>
              <LandingPage onEnterConsole={() => setView('console')} />
            </Suspense>
          )}
          <ErrorBoundary>
            <div style={{ display: view === 'console' ? 'block' : 'none' }}><Console /></div>
            {view === 'history' && <HistoryView />}
            {view === 'evaluation' && <EvaluationView prov={prov} />}
            {view === 'provenance' && <ProvenanceView prov={prov} />}
            {view === 'help' && <HelpView onStart={() => { setView('console'); setOnboard(false) }} />}
          </ErrorBoundary>
        </main>

        {view !== 'home' && (
        <footer className="border-t border-line py-5">
          <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2 px-7 text-xs text-faint">
            <span>Anvesha — agentic earth-observation intelligence. Built for ISRO PS SIH26167.</span>
            <button onClick={() => setOnboard(true)} className="underline hover:text-accent">
              How does this work?
            </button>
          </div>
        </footer>
        )}
      </div>

      {onboard && <Onboarding onDone={() => { setOnboard(false); localStorage.setItem('anvesha-onboarded', '1') }} />}
    </div>
  )
}
