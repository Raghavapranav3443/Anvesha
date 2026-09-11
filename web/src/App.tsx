import { lazy, Suspense, useEffect, useRef, useState, type ReactNode } from 'react'
import type { ComponentType } from 'react'
import Console from './components/Console'
import ProvenanceView from './components/Provenance'
import HistoryView from './components/HistoryView'
import EvaluationView from './components/EvaluationView'
import HelpView from './components/HelpView'
import BoardsView from './components/BoardsView'
import JudgeRun from './components/JudgeRun'
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

type View = 'home' | 'console' | 'history' | 'evaluation' | 'provenance' | 'help' | 'boards' | 'judge-run'

const NAV: { id: View; label: string; icon: string; hint: string }[] = [
  { id: 'console', label: 'Console', icon: 'console', hint: 'Upload imagery & ask questions' },
  { id: 'boards', label: 'Boards', icon: 'evaluation', hint: 'Mission pins over real runs' },
  { id: 'judge-run', label: 'Judge Run', icon: 'history', hint: '5-mandatory-workflow checklist' },
  { id: 'history', label: 'History', icon: 'history', hint: 'Past analyses & comparison' },
  { id: 'evaluation', label: 'Evaluation', icon: 'evaluation', hint: 'Measured benchmark scorecard' },
  { id: 'provenance', label: 'Provenance', icon: 'provenance', hint: 'Models, training data, metrics' },
  { id: 'help', label: 'Help', icon: 'help', hint: 'Glossary & how it works' },
]

/** Professional line icons (stroke inherits text color; theme-safe). */
function NavIcon({ name, className = 'h-5 w-5' }: { name: string; className?: string }) {
  const glyphs: Record<string, ReactNode> = {
    console: (
      <>
        <rect x="3" y="4" width="18" height="16" rx="2" />
        <path d="M7 9l3 3-3 3" />
        <path d="M12 15h5" />
      </>
    ),
    history: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
      </>
    ),
    evaluation: (
      <>
        <path d="M4 20h16" />
        <path d="M7 16v-5" />
        <path d="M12 16V6" />
        <path d="M17 16v-8" />
      </>
    ),
    provenance: (
      <>
        <path d="M12 3l9 5-9 5-9-5 9-5z" />
        <path d="M3 13l9 5 9-5" />
      </>
    ),
    help: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M9.6 9.4a2.5 2.5 0 1 1 3.4 2.3c-.8.35-1 .9-1 1.7" />
        <path d="M12 16.8v.2" />
      </>
    ),
    moon: <path d="M20 13.5A8 8 0 1 1 10.5 4 6.5 6.5 0 0 0 20 13.5z" />,
    sun: (
      <>
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.5 4.5l1.4 1.4M18 18l1.4 1.4M19.5 4.5L18 5.9M6 18l-1.4 1.4" />
      </>
    ),
  }
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor"
      strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      {glyphs[name] ?? null}
    </svg>
  )
}

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

const VIEW_IDS: View[] = ['home', 'console', 'history', 'evaluation', 'provenance', 'help']

export default function App() {
  // ---- restore last visited page ------------------------------------------ #
  // The active view is persisted locally so a refresh lands the user back
  // where they left off instead of on the landing page.
  const [view, setView] = useState<View>(() => {
    try {
      const saved = localStorage.getItem('anvesha-view') as View | null
      if (saved && VIEW_IDS.includes(saved) && (saved !== 'home' || hasLanding)) return saved
    } catch { /* storage unavailable */ }
    return hasLanding ? 'home' : 'console'
  })
  useEffect(() => {
    try { localStorage.setItem('anvesha-view', view) } catch { /* storage unavailable */ }
  }, [view])
  const [prov, setProv] = useState<Provenance | null>(null)
  const [theme, setTheme] = useState<'light' | 'dark'>(
    () => {
      // saved preference wins, then OS preference, then dark as baseline
      const saved = localStorage.getItem('anvesha-theme')
      if (saved === 'light' || saved === 'dark') return saved
      try {
        return window.matchMedia('(prefers-color-scheme: dark)').matches
          ? 'dark'
          : 'light'
      } catch {
        return 'dark'
      }
    })
  // ---- first-run onboarding ------------------------------------------------ #
  // Shown only the very first time the user enters the Console (never on the
  // landing page), and never again once completed/skipped.
  const [onboard, setOnboard] = useState(false)
  const onboardDismissed = useRef(false)
  useEffect(() => {
    if (view === 'console' && !onboardDismissed.current
      && !localStorage.getItem('anvesha-onboarded')) setOnboard(true)
  }, [view])

  useEffect(() => { fetchProvenance().then(setProv).catch(() => {}) }, [])

  // keep the DOM attribute and React state in sync on mount (the head script
  // pre-sets it, but React must own it from here on)
  useEffect(() => { document.documentElement.dataset.theme = theme }, [theme])

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
              <NavIcon name={n.icon} />
              <span className="text-[11px] font-medium leading-tight text-center">{n.label}</span>
            </button>
          ))}
        </nav>
        <button onClick={toggleTheme} title="Toggle light/dark theme"
          className="flex w-[60px] flex-col items-center gap-0.5 rounded-lg py-2 text-muted hover:bg-elev hover:text-body">
          <NavIcon name={theme === 'light' ? 'moon' : 'sun'} className="h-[18px] w-[18px]" />
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
            <div style={{ display: view === 'console' ? 'block' : 'none' }}><Console active={view === 'console'} /></div>
            {view === 'boards' && <BoardsView />}
            {view === 'judge-run' && <JudgeRun />}
            {view === 'history' && <HistoryView />}
            {view === 'evaluation' && <EvaluationView prov={prov} />}
            {view === 'provenance' && <ProvenanceView prov={prov} />}
            {view === 'help' && <HelpView onStart={() => { setView('console'); onboardDismissed.current = true; setOnboard(false) }} />}
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

      {onboard && <Onboarding onDone={() => { setOnboard(false); onboardDismissed.current = true; localStorage.setItem('anvesha-onboarded', '1') }} />}
    </div>
  )
}
