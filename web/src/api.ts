export interface SampleInfo {
  name: string
  file: string
  modality: string
  size: number[]
  bands: number
  georeferenced: boolean
  crs?: string | null
}

export interface TraceStep {
  step_id?: number
  name: string
  label?: string
  status?: string
  duration_ms?: number
  params?: Record<string, unknown>
  output?: Record<string, unknown>
  output_keys?: string[]
  error?: string
}

export interface JobResult {
  run_id: string
  cached?: boolean
  query: string
  selected_task: string
  answer: string
  confidence: number
  outputs: Record<string, unknown>
  execution_summary: TraceStep[]
  configuration: Record<string, unknown>
  visuals: Record<string, string>
  visual_data: Record<string, string | undefined>
  inputs?: { summary: SampleInfo; composite: string }[]
  mask_geotiff?: string
  reports: Record<string, string>
}

export interface JobState {
  job_id: string
  status: 'queued' | 'running' | 'done' | 'error'
  query: string
  trace: TraceStep[]
  result?: JobResult
  error?: string
}

const BASE = ''

export async function fetchSamples(): Promise<SampleInfo[]> {
  const r = await fetch(`${BASE}/api/samples`)
  return r.json()
}

export async function createJob(opts: {
  query: string
  taskOverride?: string
  files: File[]
  sampleNames: string[]
  dateA?: string
  dateB?: string
}): Promise<string> {
  const fd = new FormData()
  fd.append('query', opts.query)
  fd.append('task_override', opts.taskOverride ?? 'auto')
  fd.append('sample_names', opts.sampleNames.join('|'))
  fd.append('date_a', opts.dateA ?? 'T1')
  fd.append('date_b', opts.dateB ?? 'T2')
  for (const f of opts.files) fd.append('files', f)
  const r = await fetch(`${BASE}/api/jobs`, { method: 'POST', body: fd })
  if (!r.ok) throw new Error((await r.json()).detail ?? 'failed to start job')
  return (await r.json()).job_id
}

export async function pollJob(id: string): Promise<JobState> {
  const r = await fetch(`${BASE}/api/jobs/${id}`)
  return r.json()
}

export async function fetchProvenance(): Promise<Provenance> {
  const r = await fetch(`${BASE}/api/provenance`)
  return r.json()
}

export interface ModelCard {
  component: string
  file: string
  trained: boolean
  val_accuracy?: number
  label_space?: string
  synthetic?: boolean
}
export interface BenchmarkRow {
  benchmark: string
  metric: string
  n: number
  score?: number | null
  iou?: number
  f1?: number
  note?: string
}
export interface Provenance {
  models: ModelCard[]
  benchmarks: BenchmarkRow[] | null
}

/* ---------------- history ---------------- */

export interface HistoryRow {
  job_id: string
  created_at: string
  query: string
  status: string
  selected_task: string
  answer: string
  confidence: number
  run_id: string
  cached: number
}

export async function fetchHistory(limit = 30): Promise<HistoryRow[]> {
  const r = await fetch(`${BASE}/api/history?limit=${limit}`)
  return r.json()
}

export async function fetchJob(jobId: string): Promise<JobState> {
  const r = await fetch(`${BASE}/api/jobs/${jobId}`)
  return r.json()
}

/* ---------------- geo + evaluation ---------------- */

export interface GeoJSON {
  type: string
  crs?: string | null
  features: {
    type: string
    properties: { kind: string }
    geometry: { type: string; coordinates: number[][][] }
  }[]
}

export async function fetchGeo(runId: string): Promise<GeoJSON | null> {
  const r = await fetch(`${BASE}/api/geo/${runId}`)
  if (!r.ok) return null
  return r.json()
}

export async function runEvaluation(n = 300): Promise<string> {
  const r = await fetch(`${BASE}/api/evaluate/run?n=${n}`, { method: 'POST' })
  return (await r.json()).eval_id
}

export async function evalStatus(id: string): Promise<{
  status: string
  scorecard?: { results: BenchmarkRow[]; combined_normalized: number }
  error?: string
}> {
  const r = await fetch(`${BASE}/api/evaluate/status/${id}`)
  return r.json()
}
