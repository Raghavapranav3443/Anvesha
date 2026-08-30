import { useEffect, useRef, useState } from 'react'
import {
  createJob, fetchSamples, pollJob,
  type JobState, type SampleInfo,
} from '../api'
import { stepLabel as sharedStepLabel, TASK_LABELS } from '../labels'
import Results from './Results'

const EXAMPLES = [
  'Describe the land-cover and major objects visible in this image.',
  'Highlight the water body referred to in the query.',
  'What changed between these two dates, and where did the change occur?',
  'Has the built-up area increased, decreased, or remained unchanged?',
  'Use the optical and SAR images together to identify built-up and water-covered regions.',
  'Investigate urban expansion around the water body between these dates.',
]

function stepLabel(name: string): string {
  return sharedStepLabel(name)
}

function modalityBadge(m: string) {
  const styles: Record<string, string> = {
    sar: 'border-warn/40 bg-warn/10 text-warn',
    multispectral: 'border-sky-500/40 bg-sky-500/10 text-sky-600 dark:text-sky-300',
    rgb: 'border-good/40 bg-good/10 text-good',
    grayscale: 'border-line bg-elev text-muted',
  }
  return (
    <span className={`shrink-0 rounded border px-1 py-px font-mono text-[11px] uppercase tracking-wide ${styles[m] ?? styles.grayscale}`}>
      {m}
    </span>
  )
}

export function Term({ t, d }: { t: string; d: string }) {
  return (
    <span className="term" tabIndex={0}>{t}
      <span className="tip">{d}</span>
    </span>
  )
}

export function Panel({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="overflow-hidden rounded-xl border border-line bg-panel p-5 shadow-[var(--shadow-panel)]">
      <h2 className="mb-4 flex items-center gap-2 text-[14px] font-semibold uppercase tracking-[.13em] text-faint">
        {title}
        {hint && <Term t="?" d={hint} />}
      </h2>
      {children}
    </div>
  )
}

const GUIDE_TEXT = [
  'Start by adding imagery — upload your own satellite images, or click one of the demo samples below.',
  'Now ask your question in plain language — type it here, or tap one of the example questions.',
  'All set — hit Run analysis and the agent takes over: model routing, evidence overlays and an auditable trace.',
]

/** Small anchored pop-up used by the first-visit guided tour in the Console. */
function GuidePop({ step, className = '', onNext, onClose }: {
  step: number
  className?: string
  onNext: () => void
  onClose: () => void
}) {
  return (
    <div className={`fade-up absolute z-40 w-72 rounded-xl border border-accent/60 bg-panel p-3.5 shadow-2xl ${className}`}
      role="dialog" aria-label={`Guided tour step ${step + 1}`}>
      <span className="absolute -top-1.5 left-1/2 h-3 w-3 -translate-x-1/2 rotate-45 border-l border-t border-accent/60 bg-panel" />
      <div className="mb-1.5 flex items-center justify-between">
        <span className="font-mono text-[11px] uppercase tracking-[.16em] text-accent">
          Quick tour · {step + 1}/3
        </span>
        <button onClick={onClose} title="Close tour" aria-label="Close tour"
          className="px-1 text-faint transition-colors hover:text-body">×</button>
      </div>
      <p className="text-[14px] leading-snug text-muted">{GUIDE_TEXT[step]}</p>
      <div className="mt-3 flex items-center justify-between">
        <div className="flex gap-1.5">
          {[0, 1, 2].map((i) => (
            <span key={i} className={`h-1.5 w-5 rounded-full ${i === step ? 'bg-accent' : 'bg-line'}`} />
          ))}
        </div>
        <div className="flex gap-2">
          <button onClick={onClose} className="rounded-lg px-2.5 py-1 text-[13.5px] text-muted transition-colors hover:text-body">
            Close
          </button>
          <button onClick={onNext}
            className="rounded-lg bg-accent px-4 py-1 text-[13.5px] font-semibold text-white transition-colors hover:bg-accent-dim">
            {step < 2 ? 'Next' : 'Got it'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default function Console({ active = true }: { active?: boolean }) {
  const [samples, setSamples] = useState<SampleInfo[]>([])
  const [selected, setSelected] = useState<string[]>([])
  const [files, setFiles] = useState<File[]>([])
  const [query, setQuery] = useState('')
  const [override, setOverride] = useState('auto')
  const [investigate, setInvestigate] = useState(false)
  const [dateA, setDateA] = useState('T1')
  const [dateB, setDateB] = useState('T2')
  const [job, setJob] = useState<JobState | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const pollRef = useRef<number | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const lastInputs = useRef<(File | string)[]>([])

  // ---- first-visit guided tour (3 pop-ups) -------------------------------- #
  // 0: imagery upload / demo samples · 1: question bar · 2: run button.
  // Starts the first time the console becomes visible; "Close" skips the rest
  // and, like finishing the tour, marks it done so it never appears again.
  const [guideStep, setGuideStep] = useState<number | null>(null)
  const guideStarted = useRef(false)

  useEffect(() => {
    if (active && !guideStarted.current && !localStorage.getItem('anvesha-guide-done')) {
      guideStarted.current = true
      setGuideStep(0)
    }
  }, [active])

  function closeGuide() {
    setGuideStep(null)
    localStorage.setItem('anvesha-guide-done', '1')
  }

  function advanceGuide() {
    if (guideStep === null) return
    if (guideStep >= 2) closeGuide()
    else setGuideStep(guideStep + 1)
  }

  useEffect(() => { fetchSamples().then(setSamples).catch(() => {}) }, [])
  useEffect(() => () => {
    if (pollRef.current) window.clearInterval(pollRef.current)
    if (abortRef.current) abortRef.current.abort()
  }, [])

  function toggleSample(name: string) {
    setSelected((s) => s.includes(name)
      ? s.filter((x) => x !== name)
      : [...s, name].slice(0, 2))
    if (guideStep === 0) advanceGuide()
  }

  function addFiles(f: File[]) {
    setFiles(f)
    if (guideStep === 0 && f.length > 0) advanceGuide()
  }

  function onQueryChange(v: string) {
    setQuery(v)
    if (guideStep === 1 && v.trim()) advanceGuide()
  }

  async function launch(queryText: string) {
    // Cancel any previous in-flight poll
    if (pollRef.current) window.clearInterval(pollRef.current)
    if (abortRef.current) abortRef.current.abort()
    const useFiles = lastInputs.current.filter((x): x is File => x instanceof File)
    const useSamples = lastInputs.current.filter((x): x is string => typeof x === 'string')
    const effOverride = investigate && override === 'auto' ? 'investigation' : override
    setBusy(true); setError(''); setJob(null)
    try {
      const id = await createJob({
        query: queryText, taskOverride: effOverride,
        files: useFiles, sampleNames: useSamples, dateA, dateB,
      })
      abortRef.current = new AbortController()
      pollRef.current = window.setInterval(async () => {
        try {
          const st = await pollJob(id, abortRef.current?.signal)
          setJob(st)
          if (st.status === 'done' || st.status === 'error') {
            if (pollRef.current) window.clearInterval(pollRef.current)
            setBusy(false)
            if (st.status === 'error') setError(st.error?.split('\n')[0] ?? 'analysis failed')
          }
        } catch (e) {
          if (abortRef.current?.signal.aborted) return
          if (pollRef.current) window.clearInterval(pollRef.current)
          setBusy(false)
          setError(String(e))
        }
      }, 450)
    } catch (e) {
      setError(String(e)); setBusy(false)
    }
  }

  async function run() {
    if (busy || (!files.length && !selected.length)) return
    lastInputs.current = [...files, ...selected]
    await launch(query)
  }

  async function followUp(q: string) {
    if (busy) return
    setQuery(q)
    await launch(q)
  }

  function reset() {
    if (pollRef.current) window.clearInterval(pollRef.current)
    if (abortRef.current) abortRef.current.abort()
    setJob(null); setBusy(false); setError(''); setQuery(''); setSelected([]); setFiles([])
  }

  const nInputs = files.length + selected.length
  const result = job?.status === 'done' ? job.result : undefined
  const trace = job?.trace ?? []
  const suggestions: string[] = (result?.outputs?.suggestions as string[]) ?? []

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[370px_1fr]">
      {/* -------- input panel -------- */}
      <section className="fade-up space-y-4">
        <div className="relative">
          {guideStep === 0 && (
            <div className="pointer-events-none absolute -inset-1 z-30 rounded-2xl ring-2 ring-accent" />
          )}
          <Panel title="1 · Bring your imagery"
            hint="GeoTIFF/TIFF keep their geographic reference. PNG/JPEG are for benchmark datasets. Pairs must cover the same area.">
            <Dropzone files={files} onChange={addFiles} />
          <div className="mt-4 mb-1.5 text-xs font-medium uppercase tracking-wider text-faint">
            Or load a demo sample
          </div>
          <div className="grid gap-1.5">
            {samples.map((s) => {
              const on = selected.includes(s.name)
              return (
                <button key={s.name} onClick={() => toggleSample(s.name)}
                  className={`flex items-center gap-1.5 overflow-hidden rounded-lg border px-2.5 py-1.5 text-left transition-colors ${
                    on ? 'border-accent/60 bg-accent-soft'
                       : 'border-line bg-panel hover:border-muted/50'}`}>
                  <span className={`h-2 w-2 shrink-0 rounded-full ${on ? 'bg-accent' : 'bg-line'}`} />
                  <span className="min-w-0 flex-1 truncate text-[13.5px] text-body">{s.name}</span>
                  {modalityBadge(s.modality)}
                  <span className="shrink-0 font-mono text-[12px] text-faint">{s.bands}b</span>
                </button>
              )
            })}
          </div>
          {nInputs === 2 && (
            <div className="mt-4 grid grid-cols-2 gap-2">
              <Field label={<Term t="Date A" d="Label for the first (earlier) image, e.g. 'Jan 2023'." />} value={dateA} onChange={setDateA} />
              <Field label={<Term t="Date B" d="Label for the second (later) image." />} value={dateB} onChange={setDateB} />
            </div>
          )}
          </Panel>
          {guideStep === 0 && (
            <GuidePop step={0} className="left-1/2 top-full mt-3 -translate-x-1/2"
              onNext={advanceGuide} onClose={closeGuide} />
          )}
        </div>

        <Panel title="2 · Choose how to analyse"
          hint="Auto routing lets the agent decide. Investigation Mode runs a full multi-step workflow with quantified findings.">
          <div className={`mb-3 flex items-center justify-between rounded-lg border px-3 py-2.5 ${
            investigate ? 'border-accent/60 bg-accent-soft' : 'border-line bg-panel'}`}>
            <div>
              <div className="text-[15px] font-semibold text-body">🛰️ Investigation Mode</div>
              <div className="text-[13.5px] text-muted">Multi-step agent: change → water → impact → ranked zones</div>
            </div>
            <button role="switch" aria-checked={investigate}
              onClick={() => { setInvestigate(v => !v); if (!investigate) setOverride('auto') }}
              className={`relative h-6 w-11 rounded-full transition-colors ${investigate ? 'bg-accent' : 'bg-line'}`}>
              <span className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-all ${investigate ? 'left-[22px]' : 'left-0.5'}`} />
            </button>
          </div>
          {!investigate && (
            <select value={override} onChange={(e) => setOverride(e.target.value)}
              className="w-full rounded-lg border border-line bg-panel px-3 py-2 text-sm text-body outline-none focus:border-accent/60">
              <option value="auto">Auto — agentic routing</option>
              {Object.entries(TASK_LABELS).filter(([t]) => t !== 'investigation').map(([t, label]) => (
                <option key={t} value={t}>{label}</option>
              ))}
            </select>
          )}
          {investigate && nInputs !== 2 && (
            <p className="mt-2 text-[14px] text-warn">Investigation Mode needs two images (bi-temporal pair).</p>
          )}
        </Panel>
      </section>

      {/* -------- work panel -------- */}
      <section className="fade-up space-y-6" style={{ animationDelay: '.08s' }}>
        <Panel title="3 · Ask your question"
          hint="Plain language works best. The agent handles the remote-sensing vocabulary for you.">
          <div className="relative">
            {guideStep === 1 && (
              <div className="pointer-events-none absolute -inset-1 z-30 rounded-2xl ring-2 ring-accent" />
            )}
            <textarea value={query} onChange={(e) => onQueryChange(e.target.value)} rows={2}
              placeholder='e.g. "What changed between these two dates?"'
              className="w-full resize-none rounded-lg border border-line bg-panel px-4 py-3 text-[17px] text-body outline-none placeholder:text-faint focus:border-accent/60" />
            <div className="mt-3 flex flex-wrap items-center gap-2">
              {EXAMPLES.map((ex) => (
                <button key={ex} onClick={() => onQueryChange(ex)}
                  className="max-w-full truncate rounded-full border border-line bg-panel px-3 py-1 text-xs text-muted transition-colors hover:border-accent/50 hover:text-accent">
                  {ex.length > 54 ? ex.slice(0, 54) + '…' : ex}
                </button>
              ))}
            </div>
            {guideStep === 1 && (
              <GuidePop step={1} className="left-1/2 top-full mt-3 -translate-x-1/2"
                onNext={advanceGuide} onClose={closeGuide} />
            )}
          </div>
          <div className="mt-4 flex items-center justify-between">
            <span className="font-mono text-[13.5px] text-faint">
              {nInputs} input{nInputs === 1 ? '' : 's'}
              {investigate && ' · investigation mode'}
            </span>
            <div className="flex items-center gap-2">
              <div className="relative">
                {guideStep === 2 && (
                  <div className="pointer-events-none absolute -inset-1 z-30 rounded-xl ring-2 ring-accent" />
                )}
                <button onClick={run} disabled={busy || !nInputs || (investigate && nInputs !== 2)}
                  className="relative overflow-hidden rounded-lg bg-accent px-7 py-2 text-[16.5px] font-semibold text-white transition-all hover:bg-accent-dim disabled:cursor-not-allowed disabled:opacity-40">
                  {busy ? 'Analysing…' : investigate ? 'Run investigation' : 'Run analysis'}
                  {busy && <span className="scanning absolute inset-0" />}
                </button>
                {guideStep === 2 && (
                  <GuidePop step={2} className="bottom-full right-0 mb-3"
                    onNext={advanceGuide} onClose={closeGuide} />
                )}
              </div>
              {(job || busy) && (
                <button onClick={reset} disabled={busy}
                  className="rounded-lg border border-line px-4 py-2 text-[14px] font-medium text-muted transition-colors hover:border-accent/50 hover:text-accent disabled:opacity-40">
                  Reset
                </button>
              )}
            </div>
          </div>
          {error && (
            <div className="mt-3 rounded-lg border border-bad/40 bg-bad/10 px-3 py-2 text-sm text-bad">{error}</div>
          )}
        </Panel>

        {(trace.length > 0 || busy) && (
          <TraceTimeline trace={trace} running={!!busy && job?.status !== 'done'} />
        )}

        {result && (
          <>
            <Results result={result} onFollowUp={followUp} />
            {suggestions.length > 0 && (
              <Panel title="Ask the data back" hint="One click launches a follow-up analysis on the same imagery.">
                <div className="flex flex-wrap gap-2">
                  {suggestions.map(q => (
                    <button key={q} onClick={() => followUp(q)} disabled={busy}
                      className="rounded-full border border-accent/40 bg-accent-soft px-3.5 py-1.5 text-[14.5px] text-accent transition-colors hover:bg-accent/20 disabled:opacity-40">
                      {q} →
                    </button>
                  ))}
                </div>
              </Panel>
            )}
          </>
        )}
      </section>
    </div>
  )
}

/* ------------------------------------------------------------------ */

function Field({ label, value, onChange }: { label: React.ReactNode; value: string; onChange: (v: string) => void }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[13.5px] text-muted">{label}</span>
      <input value={value} onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-line bg-panel px-3 py-1.5 font-mono text-sm text-body outline-none focus:border-accent/60" />
    </label>
  )
}

function Dropzone({ files, onChange }: { files: File[]; onChange: (f: File[]) => void }) {
  const ref = useRef<HTMLInputElement>(null)
  const [hot, setHot] = useState(false)
  return (
    <div
      onClick={() => ref.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setHot(true) }}
      onDragLeave={() => setHot(false)}
      onDrop={(e) => { e.preventDefault(); setHot(false); onChange([...files, ...Array.from(e.dataTransfer.files)].slice(0, 2)) }}
      className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed px-4 py-6 transition-colors ${
        hot ? 'border-accent bg-accent-soft' : 'border-muted/40 hover:border-muted'}`}>
      <svg viewBox="0 0 24 24" className="mb-2 h-7 w-7 stroke-faint" fill="none" strokeWidth="1.5">
        <path d="M12 16V4m0 0l-4 4m4-4l4 4M4 17v2a1 1 0 001 1h14a1 1 0 001-1v-2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <span className="text-sm text-muted">Drop GeoTIFF / PNG / JPEG here</span>
      <span className="mt-0.5 text-[13.5px] text-faint">up to 2 images · single image or a pair</span>
      <input ref={ref} type="file" multiple accept=".tif,.tiff,.png,.jpg,.jpeg" className="hidden"
        onChange={(e) => onChange(Array.from(e.target.files ?? []).slice(0, 2))} />
      {files.length > 0 && (
        <ul className="mt-3 w-full space-y-1">
          {files.map((f) => (
            <li key={f.name} className="flex items-center justify-between rounded bg-elev px-2 py-1 text-xs text-muted">
              <span className="truncate">{f.name}</span>
              <button onClick={(e) => { e.stopPropagation(); onChange(files.filter((x) => x !== f)) }}
                className="px-1 text-faint hover:text-bad">×</button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export function TraceTimeline({ trace, running }: { trace: JobState['trace']; running: boolean }) {
  const steps = trace.filter((s) => s.name !== 'finish')
  return (
    <Panel title="Agent execution trace"
      hint="Every step the agent took, with parameters and timing — your audit trail.">
      <ol className="relative ml-2 space-y-4 border-l border-line pl-6">
        {steps.map((s, i) => {
          const last = i === steps.length - 1
          const active = running && last
          const ok = !running || !last
          return (
            <li key={i} className={`relative ${active ? 'step-active' : ''}`}>
              <span className={`absolute -left-[31px] top-1 h-2.5 w-2.5 rounded-full ${
                active ? 'bg-accent shadow-[0_0_8px_var(--c-accent)]' : ok ? 'bg-good' : 'bg-line'}`} />
              <div className="flex items-baseline justify-between gap-3">
                <span className="font-mono text-[14.5px] text-body">
                  {s.label ?? stepLabel(s.name)}
                </span>
                {s.duration_ms != null && (
                  <span className="font-mono text-[13px] text-faint">{s.duration_ms} ms</span>
                )}
              </div>
              {s.output_keys && (
                <div className="mt-0.5 font-mono text-[13px] text-faint">→ {s.output_keys.join(', ')}</div>
              )}
              {s.params && Object.keys(s.params).length > 0 && (
                <details className="mt-1">
                  <summary className="cursor-pointer font-mono text-[13px] text-faint hover:text-muted">params</summary>
                  <pre className="mt-1 overflow-auto rounded bg-elev p-2 font-mono text-[13px] text-muted">
                    {JSON.stringify(s.params, null, 1)}
                  </pre>
                </details>
              )}
            </li>
          )
        })}
        {running && steps.length > 0 && (
          <li className="relative">
            <span className="absolute -left-[31px] top-1 h-2.5 w-2.5 animate-pulse rounded-full bg-line" />
            <span className="font-mono text-[14.5px] text-faint">working…</span>
          </li>
        )}
      </ol>
    </Panel>
  )
}
