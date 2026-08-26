import { useEffect, useState } from 'react'
import Console from './components/Console'
import ProvenanceView from './components/Provenance'
import HistoryView from './components/HistoryView'
import EvaluationView from './components/EvaluationView'
import HelpView from './components/HelpView'
import Onboarding from './components/Onboarding'
import { fetchProvenance, type Provenance } from './api'

type View = 'console' | 'history' | 'evaluation' | 'provenance' | 'help'

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
  const [view, setView] = useState<View>('console')
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
    <div className="flex min-h-full">
      {/* ---- left rail nav ---- */}
      <aside className="fixed inset-y-0 left-0 z-40 flex w-[76px] flex-col items-center border-r border-line bg-panel py-4">
        <div className="mb-6"><OrbitMark /></div>
        <nav className="flex flex-1 flex-col gap-1.5">
          {NAV.map(n => (
            <button key={n.id} onClick={() => setView(n.id)} title={n.hint}
              className={`group flex w-[60px] flex-col items-center gap-0.5 rounded-lg py-2 transition-colors ${
                view === n.id ? 'bg-accent-soft text-accent' : 'text-muted hover:bg-elev hover:text-body'}`}>
              <span className="text-[18px] leading-none">{n.icon}</span>
              <span className="text-[12px] font-medium">{n.label}</span>
            </button>
          ))}
        </nav>
        <button onClick={toggleTheme} title="Toggle light/dark theme"
          className="flex w-[60px] flex-col items-center gap-0.5 rounded-lg py-2 text-muted hover:bg-elev hover:text-body">
          <span className="text-[16px]">{theme === 'light' ? '🌙' : '☀️'}</span>
          <span className="text-[12px]">{theme === 'light' ? 'Dark' : 'Light'}</span>
        </button>
      </aside>

      {/* ---- main column ---- */}
      <div className="ml-[76px] flex min-h-full flex-1 flex-col">
        <header className="sticky top-0 z-30 border-b border-line bg-panel/90 backdrop-blur">
          <div className="relative flex h-[68px] items-center gap-4 overflow-hidden px-7">
            {/* signature animation: satellite pass */}
            <span className="sat-pass pointer-events-none absolute inset-x-0 top-1/2 h-px bg-gradient-to-r from-transparent via-accent/40 to-transparent" />
            <div className="relative z-10 mr-auto">
              <div className="text-[19px] font-bold tracking-tight text-body">
                Anvesha
                <span className="ml-2 rounded border border-accent/40 bg-accent-soft px-1.5 py-px align-middle font-mono text-[10px] font-medium uppercase tracking-wide text-accent">
                  EO Investigation System
                </span>
              </div>
              <div className="-mt-0.5 font-mono text-[12px] uppercase tracking-[.16em] text-faint">
                Earth Observation &amp; Investigation System · SIH26167
              </div>
            </div>
            <span className="hidden items-center gap-1.5 rounded-full border border-line px-3 py-1 text-[12px] text-muted md:flex">
              <span className="h-2 w-2 rounded-full bg-good" /> system ready
            </span>
          </div>
        </header>

        <main className="mx-auto w-full max-w-7xl flex-1 px-7 py-8">
          {view === 'console' && <Console />}
          {view === 'history' && <HistoryView />}
          {view === 'evaluation' && <EvaluationView prov={prov} />}
          {view === 'provenance' && <ProvenanceView prov={prov} />}
          {view === 'help' && <HelpView onStart={() => { setView('console'); setOnboard(false) }} />}
        </main>

        <footer className="border-t border-line py-5">
          <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2 px-7 text-xs text-faint">
            <span>Anvesha — agentic earth-observation intelligence. Built for ISRO PS SIH26167.</span>
            <button onClick={() => setOnboard(true)} className="underline hover:text-accent">
              How does this work?
            </button>
          </div>
        </footer>
      </div>

      {onboard && <Onboarding onDone={() => { setOnboard(false); localStorage.setItem('anvesha-onboarded', '1') }} />}
    </div>
  )
}
