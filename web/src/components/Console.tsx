import { useEffect, useRef, useState } from 'react'
import {
  createJob, fetchSamples, pollJob,
  type JobState, type SampleInfo,
} from '../api'
import { stepLabel as sharedStepLabel, TASK_LABELS, taskLabel } from '../labels'
import { SETUPS } from '../setups'
import AcquirePanel from './AcquirePanel'
import Results, { ExportLinks } from './Results'

const EXAMPLES = [
  'Investigate urban expansion around the water body between these dates.',
  'Describe the land-cover and major objects visible in this image.',
  'Highlight the water body referred to in the query.',
  'What changed between these two dates, and where did the change occur?',
  'Has the built-up area increased, decreased, or remained unchanged?',
  'Use the optical and SAR images together to identify built-up and water-covered regions.',
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
    <span className={`shrink-0 rounded border px-1 py-px font-mono text-xs uppercase tracking-wide ${styles[m] ?? styles.grayscale}`}>
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
      <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-[.13em] text-faint">
        {title}
        {hint && <Term t="?" d={hint} />}
      </h2>
      {children}
    </div>
  )
}

const GUIDE_TEXT = [
  'Start by adding imagery — upload your own satellite images here, or pick a sample in the Demo inputs box below.',
  'Now ask your question in plain language — type it in the bar floating at the bottom of your screen, or tap an example in the Demo inputs box.',
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
        <span className="font-mono text-xs uppercase tracking-[.16em] text-accent">
          Quick tour · {step + 1}/3
        </span>
        <button onClick={onClose} title="Close tour" aria-label="Close tour"
          className="px-1 text-faint transition-colors hover:text-body">×</button>
      </div>
      <p className="text-sm leading-snug text-muted">{GUIDE_TEXT[step]}</p>
      <div className="mt-3 flex items-center justify-between">
        <div className="flex gap-1.5">
          {[0, 1, 2].map((i) => (
            <span key={i} className={`h-1.5 w-5 rounded-full ${i === step ? 'bg-accent' : 'bg-line'}`} />
          ))}
        </div>
        <div className="flex gap-2">
          <button onClick={onClose} className="rounded-lg px-2.5 py-1 text-sm text-muted transition-colors hover:text-body">
            Close
          </button>
          <button onClick={onNext}
            className="rounded-lg bg-accent px-4 py-1 text-sm font-semibold text-white transition-colors hover:bg-accent-dim">
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
  const [modality, setModality] = useState<'auto' | 'sar' | 'optical'>('auto')
  const [job, setJob] = useState<JobState | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [previewOpen, setPreviewOpen] = useState(false)  // chosen-images popup
  const [outputOpen, setOutputOpen] = useState(false)    // report popup
  const pollRef = useRef<number | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const lastInputs = useRef<(File | string)[]>([])
  // Imagery fetched by the online layer. Kept in a ref as well as state because
  // `launch` is called immediately after a fetch resolves, before React has
  // re-rendered, and a stale closure would send the job off with no images.
  const acquireRef = useRef<string>('')
  const [acquireId, setAcquireId] = useState('')

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

  // Object URLs for in-browser preview of chosen files (PNG/JPEG only —
  // GeoTIFFs get a server-rendered or placeholder card in the preview popup).
  const [fileUrls, setFileUrls] = useState<{ name: string; url: string | null }[]>([])
  useEffect(() => {
    const next = files.map((f) => ({
      name: f.name,
      url: /^image\/(png|jpe?g|webp)$/.test(f.type) ? URL.createObjectURL(f) : null,
    }))
    setFileUrls(next)
    return () => next.forEach((x) => { if (x.url) URL.revokeObjectURL(x.url) })
  }, [files])

  /** Choosing imagery by hand replaces anything fetched online, so a job never
   *  ends up carrying both (the server caps an analysis at two images). */
  function clearAcquired() {
    if (acquireRef.current) {
      acquireRef.current = ''
      setAcquireId('')
    }
  }

  function toggleSample(name: string) {
    clearAcquired()
    setSelected((s) => s.includes(name)
      ? s.filter((x) => x !== name)
      : [...s, name].slice(0, 2))
    if (guideStep === 0) advanceGuide()
  }

  function addFiles(f: File[]) {
    clearAcquired()
    setFiles(f)
    if (guideStep === 0 && f.length > 0) advanceGuide()
  }

  function runSetup(s: typeof SETUPS[number]) {
    setSelected(s.sampleNames.filter((n) => samples.some((x) => x.name === n)))
    setQuery(s.query)
    setGuideStep(null)
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
        modality,
        acquireId: acquireRef.current || undefined,
      })
      abortRef.current = new AbortController()
      pollRef.current = window.setInterval(async () => {
        try {
          const st = await pollJob(id, abortRef.current?.signal)
          setJob(st)
          if (st.status === 'done' || st.status === 'error') {
            if (pollRef.current) window.clearInterval(pollRef.current)
            setBusy(false)
            if (st.status === 'done') setOutputOpen(true)   // pop the report
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
    if (busy || (!files.length && !selected.length && !acquireRef.current)) return
    lastInputs.current = [...files, ...selected]
    await launch(query)
  }

  /** Online imagery arrived: analyse it straight away, in one click. */
  function onAcquired(id: string) {
    acquireRef.current = id
    setAcquireId(id)
    lastInputs.current = []            // fetched imagery replaces any selection
    setFiles([])
    setSelected([])
    const q = query.trim() || 'What changed between these two dates?'
    setQuery(q)
    void launch(q)
  }

  async function followUp(q: string) {
    if (busy) return
    setQuery(q)
    await launch(q)
  }

  function reset() {
    if (pollRef.current) window.clearInterval(pollRef.current)
    if (abortRef.current) abortRef.current.abort()
    acquireRef.current = ''
    setAcquireId('')
    setJob(null); setBusy(false); setError(''); setQuery(''); setSelected([]); setFiles([])
    setOutputOpen(false); setPreviewOpen(false)
  }

  const nInputs = files.length + selected.length
  const result = job?.status === 'done' ? job.result : undefined
  const trace = job?.trace ?? []
  const suggestions: string[] = (result?.outputs?.suggestions as string[]) ?? []

  return (
    <div className="flex flex-col gap-6 pb-44">
      {/* -------- online: fetch the imagery for the user -------- */}
      <AcquirePanel onAcquired={onAcquired} busy={busy} />

      {/* -------- row 1: upload + analyse, side by side -------- */}
      <section className="fade-up grid grid-cols-1 items-start gap-6 lg:grid-cols-2">
        <div className="relative">
          {guideStep === 0 && (
            <div className="pointer-events-none absolute -inset-1 z-30 rounded-2xl ring-2 ring-accent" />
          )}
          <Panel title="1 · Bring your imagery"
            hint={acquireId
              ? `Online imagery selected (acquisition ${acquireId}). Uploading files replaces it.`
              : "GeoTIFF/TIFF keep their geographic reference. PNG/JPEG are for benchmark datasets. Pairs must cover the same area."}>
            <Dropzone files={files} onChange={addFiles} />
            {nInputs > 0 && (
              <button onClick={() => setPreviewOpen(true)}
                className="mt-3 flex w-full items-center gap-2 rounded-lg border border-line bg-elev px-3 py-2 text-left transition-colors hover:border-accent/50">
                {fileUrls.filter((x) => x.url).slice(0, 2).map((f) => (
                  <img key={f.name} src={f.url!} alt={f.name}
                    className="h-10 w-10 rounded border border-line object-cover" />
                ))}
                {selected.slice(0, 2).map((name) => (
                  <img key={name} src={`/api/samples/${encodeURIComponent(name)}/preview`} alt={name}
                    className="h-10 w-10 rounded border border-line object-cover" />
                ))}
                <span className="text-sm text-accent">
                  Preview {nInputs} chosen image{nInputs === 1 ? '' : 's'} ↗
                </span>
              </button>
            )}
          {nInputs === 2 && (
            <div className="mt-4 grid grid-cols-2 gap-2">
              <Field label={<Term t="Date A" d="Label for the first (earlier) image, e.g. 'Jan 2023'." />} value={dateA} onChange={setDateA} />
              <Field label={<Term t="Date B" d="Label for the second (later) image." />} value={dateB} onChange={setDateB} />
            </div>
          )}
          {nInputs >= 1 && (
            <div className="mt-3 flex items-center gap-2">
              <span className="shrink-0 font-mono text-xs uppercase tracking-wider text-faint">Sensor</span>
              {(['auto', 'sar', 'optical'] as const).map((m) => (
                <button key={m} onClick={() => setModality(m)}
                  className={`rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors ${
                    modality === m
                      ? 'border-accent/60 bg-accent-soft text-accent'
                      : 'border-line bg-panel text-muted hover:border-accent/50 hover:text-accent'}`}>
                  {m}
                </button>
              ))}
              {modality !== 'auto' && (
                <span className="text-xs text-faint">
                  {modality === 'sar' ? 'forced SAR' : 'forced optical'} — the stats badge overrides if it disagrees
                </span>
              )}
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
              <div className="text-sm font-semibold text-body">🛰️ Investigation Mode</div>
              <div className="text-sm text-muted">Multi-step agent: change → water → impact → ranked zones</div>
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
            <p className="mt-2 text-sm text-warn">Investigation Mode needs two images (bi-temporal pair).</p>
          )}
        </Panel>
      </section>

      {/* -------- row 2: Demo Inputs — static panel, same style as the boxes above -------- */}
      <Panel title="Demo inputs"
        hint="One-click setups pair demo imagery with its question; samples and example questions also work à la carte.">
        <DemoInputs samples={samples} selected={selected} toggleSample={toggleSample}
          runSetup={runSetup} onQueryChange={onQueryChange}
          onPreview={() => setPreviewOpen(true)} />
      </Panel>

      {/* -------- row 3: pipeline (agent execution trace) + follow-ups -------- */}
      <section className="space-y-6">

        {(trace.length > 0 || busy) && (
          <TraceTimeline trace={trace} running={!!busy && job?.status !== 'done'} />
        )}

        {suggestions.length > 0 && (
              <Panel title="Ask the data back" hint="One click launches a follow-up analysis on the same imagery.">
                <div className="flex flex-wrap gap-2">
                  {suggestions.map(q => (
                    <button key={q} onClick={() => followUp(q)} disabled={busy}
                      className="rounded-full border border-accent/40 bg-accent-soft px-3.5 py-1.5 text-sm text-accent transition-colors hover:bg-accent/20 disabled:opacity-40">
                      {q} →
                    </button>
                  ))}
                </div>
              </Panel>
        )}
      </section>

      {/* -------- floating query bar: fixed to the bottom of the user's screen -------- */}
      <div className="pointer-events-none fixed inset-x-0 bottom-0 z-40 flex justify-center px-4 pb-4">
        <div className="pointer-events-auto w-full max-w-3xl rounded-2xl border border-line bg-panel p-3 shadow-[0_12px_40px_rgba(10,16,28,0.45)]">
          <div className="flex flex-wrap items-end gap-3">
            <div className="relative min-w-0 flex-1">
              {guideStep === 1 && (
                <div className="pointer-events-none absolute -inset-1 z-30 rounded-lg ring-2 ring-accent" />
              )}
              <textarea value={query} onChange={(e) => onQueryChange(e.target.value)} rows={2}
                placeholder='e.g. "What changed between these two dates?"'
                className="w-full resize-none rounded-lg border border-line bg-panel px-4 py-3 text-base text-body outline-none placeholder:text-faint focus:border-accent/60" />
              {guideStep === 1 && (
                <GuidePop step={1} className="bottom-full left-0 mb-3"
                  onNext={advanceGuide} onClose={closeGuide} />
              )}
            </div>
            <div className="relative shrink-0">
              {guideStep === 2 && (
                <div className="pointer-events-none absolute -inset-1 z-30 rounded-xl ring-2 ring-accent" />
              )}
              <button onClick={run} disabled={busy || !nInputs || (investigate && nInputs !== 2)}
                className="relative overflow-hidden rounded-lg bg-accent px-7 py-2.5 text-base font-semibold text-white transition-all hover:bg-accent-dim disabled:cursor-not-allowed disabled:opacity-40">
                {busy ? 'Analysing…' : investigate ? 'Run investigation' : 'Run analysis'}
                {busy && <span className="scanning absolute inset-0" />}
              </button>
              {guideStep === 2 && (
                <GuidePop step={2} className="bottom-full right-0 mb-3"
                  onNext={advanceGuide} onClose={closeGuide} />
              )}
            </div>
            {result && !busy && (
              <button onClick={() => setOutputOpen(true)}
                className="shrink-0 rounded-lg border border-accent/50 bg-accent-soft px-4 py-2.5 text-sm font-semibold text-accent transition-colors hover:bg-accent/30">
                View output
              </button>
            )}
            {(job || busy) && (
              <button onClick={reset} disabled={busy}
                className="shrink-0 rounded-lg border border-line px-4 py-2 text-sm font-medium text-muted transition-colors hover:border-accent/50 hover:text-accent disabled:opacity-40">
                Reset
              </button>
            )}
          </div>
          <div className="mt-2 px-1">
            {error
              ? <span className="text-sm text-bad">{error}</span>
              : <span className="font-mono text-xs text-faint">
                  {nInputs} input{nInputs === 1 ? '' : 's'}{investigate && ' · investigation mode'}
                </span>}
          </div>
        </div>
      </div>

      {/* -------- chosen-images preview popup -------- */}
      {previewOpen && (
        <Modal title="Chosen images" subtitle="Inputs for the next run"
          onClose={() => setPreviewOpen(false)}>
          <div className="space-y-4">
            {fileUrls.map((f) => (
              <figure key={f.name}>
                {f.url
                  ? <img src={f.url} alt={f.name}
                      className="max-h-[50vh] w-full rounded-lg border border-line object-contain" />
                  : <div className="grid h-40 place-items-center rounded-lg border border-line bg-elev text-sm text-faint">
                      GeoTIFF — no in-browser preview; rendered during analysis
                    </div>}
                <figcaption className="mt-1 font-mono text-xs text-muted">{f.name}</figcaption>
              </figure>
            ))}
            {selected.map((name) => (
              <figure key={name}>
                <img src={`/api/samples/${encodeURIComponent(name)}/preview`} alt={name}
                  className="max-h-[50vh] w-full rounded-lg border border-line object-contain" />
                <figcaption className="mt-1 font-mono text-xs text-muted">{name} · demo sample</figcaption>
              </figure>
            ))}
            {nInputs === 0 && (
              <p className="text-sm text-muted">No images chosen yet — upload files or pick demo images in the Demo inputs box.</p>
            )}
          </div>
        </Modal>
      )}

      {/* -------- output popup (report; retained until Reset) -------- */}
      {outputOpen && result && (
        <Modal title="Analysis output"
          subtitle={`${taskLabel(result.selected_task)} · run ${result.run_id}`}
          onClose={() => setOutputOpen(false)}
          headerRight={<ExportLinks runId={result.run_id} task={result.selected_task} />}>
          <Results result={result} onFollowUp={followUp} hideExports
            onRequestSwitch={(m) => { setModality(m) }} />
        </Modal>
      )}
    </div>
  )
}

/** Small reusable popup window: closable (backdrop click / Escape / ×),
 *  scrollable body, optional top-right header actions. */
function Modal({ title, subtitle, onClose, headerRight, children }: {
  title: string
  subtitle?: string
  onClose: () => void
  headerRight?: React.ReactNode
  children: React.ReactNode
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
      role="dialog" aria-modal="true" aria-label={title}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="fade-up relative flex max-h-[85vh] w-full max-w-4xl flex-col overflow-hidden rounded-xl border border-line bg-panel shadow-2xl">
        <div className="flex items-center gap-3 border-b border-line px-4 py-3">
          <div className="min-w-0 flex-1">
            <div className="truncate text-sm font-semibold text-body">{title}</div>
            {subtitle && <div className="truncate font-mono text-xs text-faint">{subtitle}</div>}
          </div>
          {headerRight}
          <button onClick={onClose} title="Close" aria-label="Close"
            className="shrink-0 rounded-lg px-2 py-1 text-lg text-faint transition-colors hover:bg-elev hover:text-body">×</button>
        </div>
        <div className="flex-1 overflow-y-auto p-5">{children}</div>
      </div>
    </div>
  )
}

/** "Demo inputs" panel body (wrapped in a standard Panel by the Console):
 *  paired setups, example questions and demo images in one place. The
 *  investigation setup is highlighted (judge-proofing: the workflow is
 *  otherwise undiscoverable). */
function DemoInputs({ samples, selected, toggleSample, runSetup, onQueryChange,
  onPreview }: {
  samples: SampleInfo[]
  selected: string[]
  toggleSample: (name: string) => void
  runSetup: (s: typeof SETUPS[number]) => void
  onQueryChange: (v: string) => void
  onPreview: () => void
}) {
  return (
    <>
      <div className="grid gap-3 lg:grid-cols-2">
        <div>
          <div className="mb-1.5 text-xs text-muted">Paired demo setups</div>
          <div className="flex flex-wrap gap-1.5">
            {SETUPS.map((s) => {
              const inv = s.id === 'investigation'
              return (
                <button key={s.id} onClick={() => runSetup(s)}
                  title={`${s.title} — exercises ${s.patches.join(', ')}`}
                  className={`max-w-full truncate rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                    inv
                      ? 'border border-accent bg-accent text-white hover:bg-accent-dim'
                      : 'border border-accent/40 bg-accent-soft/50 text-accent hover:border-accent hover:bg-accent-soft'}`}>
                  {inv ? '★ ' : '▸ '}{s.title}
                </button>
              )
            })}
          </div>
        </div>
        <div>
          <div className="mb-1.5 text-xs text-muted">Demo questions</div>
          <div className="flex flex-wrap gap-1.5">
            {EXAMPLES.map((ex) => (
              <button key={ex} onClick={() => onQueryChange(ex)}
                className="max-w-full truncate rounded-full border border-line bg-panel px-3 py-1 text-xs text-muted transition-colors hover:border-accent/50 hover:text-accent">
                {ex.length > 54 ? ex.slice(0, 54) + '…' : ex}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="mt-3">
        <div className="mb-1.5 text-xs text-muted">
          Demo images <span className="text-faint">(▢ previews the sample)</span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {samples.map((s) => {
            const on = selected.includes(s.name)
            return (
              <span key={s.name}
                className={`flex items-center gap-1 overflow-hidden rounded-lg border transition-colors ${
                  on ? 'border-accent/60 bg-accent-soft' : 'border-line bg-panel hover:border-muted/50'}`}>
                <button onClick={() => toggleSample(s.name)}
                  className="flex min-w-0 items-center gap-1.5 py-1.5 pl-2.5 pr-1">
                  <span className={`h-2 w-2 shrink-0 rounded-full ${on ? 'bg-accent' : 'bg-line'}`} />
                  <span className="min-w-0 flex-1 truncate text-sm text-body">{s.name}</span>
                </button>
                <button onClick={onPreview} title={`Preview ${s.name}`} aria-label={`Preview ${s.name}`}
                  className="shrink-0 border-l border-line px-2 py-1.5 text-xs text-faint hover:text-accent">▢</button>
                {modalityBadge(s.modality)}
                <span className="shrink-0 pr-2 font-mono text-xs text-faint">{s.bands}b</span>
              </span>
            )
          })}
        </div>
      </div>
    </>
  )
}

/* ------------------------------------------------------------------ */

function Field({ label, value, onChange }: { label: React.ReactNode; value: string; onChange: (v: string) => void }) {
  return (
    <label className="block">
      <span className="mb-1 block text-sm text-muted">{label}</span>
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
      <span className="mt-0.5 text-sm text-faint">up to 2 images · single image or a pair</span>
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
                <span className="font-mono text-sm text-body">
                  {s.label ?? stepLabel(s.name)}
                </span>
                {s.duration_ms != null && (
                  <span className="font-mono text-sm text-faint">{s.duration_ms} ms</span>
                )}
              </div>
              {s.output_keys && (
                <div className="mt-0.5 font-mono text-sm text-faint">→ {s.output_keys.join(', ')}</div>
              )}
              {s.params && Object.keys(s.params).length > 0 && (
                <details className="mt-1">
                  <summary className="cursor-pointer font-mono text-sm text-faint hover:text-muted">params</summary>
                  <pre className="mt-1 overflow-auto rounded bg-elev p-2 font-mono text-sm text-muted">
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
            <span className="font-mono text-sm text-faint">working…</span>
          </li>
        )}
      </ol>
    </Panel>
  )
}
