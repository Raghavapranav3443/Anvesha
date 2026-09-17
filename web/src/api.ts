// ---------------------------------------------------------------------------
// Structured API error
// ---------------------------------------------------------------------------

/**
 * Thrown by apiFetch() for any non-2xx response.
 * `status`  — HTTP status code
 * `code`    — machine-readable code from the server (e.g. "queue_full")
 * `hint`    — optional human-readable suggestion from the server
 */
export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
    public readonly code: string = 'error',
    public readonly hint: string = '',
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

// ---------------------------------------------------------------------------
// Core fetch wrapper
// ---------------------------------------------------------------------------

const BASE = ''

const DEFAULT_TIMEOUT_MS = 15_000
const MAX_RETRIES = 3
/** Status codes that are safe to retry. */
const RETRYABLE = new Set([429, 503, 502])

/**
 * Opinionated fetch wrapper with:
 * - Per-request timeout (default 15 s) via AbortSignal.timeout
 * - Automatic exponential-backoff retry on 429/502/503 (up to MAX_RETRIES)
 * - Typed ApiError for every non-2xx response
 */
async function apiFetch(
  input: string,
  init: RequestInit & { timeoutMs?: number } = {},
): Promise<Response> {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, signal: callerSignal, ...rest } = init

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    // Combine caller signal + timeout signal
    const timeoutSignal = AbortSignal.timeout(timeoutMs)
    const signal =
      callerSignal
        ? AbortSignal.any([callerSignal, timeoutSignal])
        : timeoutSignal

    let res: Response
    try {
      res = await fetch(`${BASE}${input}`, { ...rest, signal })
    } catch (err: unknown) {
      // Network failure or abort — don't retry aborts from the caller
      if (callerSignal?.aborted) throw err
      if (attempt < MAX_RETRIES) {
        await _sleep(_backoff(attempt))
        continue
      }
      const msg =
        err instanceof Error ? err.message : 'Network error'
      throw new ApiError(0, msg, 'network_error')
    }

    if (res.ok) return res

    // Retry on specific transient errors
    if (RETRYABLE.has(res.status) && attempt < MAX_RETRIES) {
      // Respect Retry-After header if present
      const retryAfter = res.headers.get('Retry-After')
      const delay = retryAfter
        ? Math.min(parseInt(retryAfter, 10) * 1000, 30_000)
        : _backoff(attempt)
      await _sleep(delay)
      continue
    }

    // Parse structured error from server
    let detail = `HTTP ${res.status}`
    let code = 'error'
    let hint = ''
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
      else if (typeof body.detail === 'object') detail = body.detail?.detail ?? detail
      if (body.code) code = String(body.code)
      if (body.hint) hint = String(body.hint)
    } catch {
      /* body wasn't JSON — keep defaults */
    }
    throw new ApiError(res.status, detail, code, hint)
  }

  // Should be unreachable
  throw new ApiError(0, 'Exceeded retry limit.', 'retry_exhausted')
}

function _sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function _backoff(attempt: number): number {
  // 500 ms, 1 s, 2 s — capped at 4 s
  return Math.min(500 * Math.pow(2, attempt), 4_000)
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface SampleInfo {
  name: string
  file: string
  modality: string
  size: number[]
  bands: number
  georeferenced: boolean
  crs?: string | null
  acquired?: string | null          // B8: acquisition date when known
  modality_certainty?: {
    label: string                    // 'sar' | 'multispectral' | 'rgb' | 'grayscale'
    confidence: number               // vote share 0..1
    votes: Record<string, string>    // filename / stats / override
  } | null                          // B1: stats-first certainty when emitted
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

export interface GeoJSON {
  type: string
  crs?: string | null
  features: {
    type: string
    properties: { kind: string }
    geometry: { type: string; coordinates: number[][][] }
  }[]
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

export async function fetchSamples(): Promise<SampleInfo[]> {
  const r = await apiFetch('/api/samples')
  return r.json()
}

export async function createJob(opts: {
  query: string
  taskOverride?: string
  files: File[]
  sampleNames: string[]
  dateA?: string
  dateB?: string
  modality?: 'auto' | 'sar' | 'optical'   // B1: explicit modality override
}): Promise<string> {
  const fd = new FormData()
  fd.append('query', opts.query)
  fd.append('task_override', opts.taskOverride ?? 'auto')
  fd.append('sample_names', opts.sampleNames.join('|'))
  fd.append('date_a', opts.dateA ?? 'T1')
  fd.append('date_b', opts.dateB ?? 'T2')
  fd.append('modality', opts.modality ?? 'auto')
  for (const f of opts.files) fd.append('files', f)
  // Allow up to 2 min for large uploads
  const r = await apiFetch('/api/jobs', { method: 'POST', body: fd, timeoutMs: 120_000 })
  return (await r.json()).job_id
}

/**
 * Poll a single job.
 * @param signal  Optional AbortSignal — pass one from the component so
 *                in-flight requests are cancelled on unmount.
 */
export async function pollJob(id: string, signal?: AbortSignal): Promise<JobState> {
  const r = await apiFetch(`/api/jobs/${id}`, { signal })
  return r.json()
}

export async function fetchProvenance(): Promise<Provenance> {
  const r = await apiFetch('/api/provenance')
  return r.json()
}

export async function fetchHistory(limit = 30): Promise<HistoryRow[]> {
  const r = await apiFetch(`/api/history?limit=${limit}`)
  return r.json()
}

export async function fetchJob(jobId: string): Promise<JobState> {
  const r = await apiFetch(`/api/jobs/${jobId}`)
  return r.json()
}

export async function fetchGeo(runId: string): Promise<GeoJSON | null> {
  try {
    const r = await apiFetch(`/api/geo/${runId}`)
    return r.json()
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) return null
    throw err
  }
}

export async function runEvaluation(n = 300): Promise<string> {
  const r = await apiFetch(`/api/evaluate/run?n=${n}`, { method: 'POST' })
  return (await r.json()).eval_id
}

export async function evalStatus(id: string): Promise<{
  status: string
  scorecard?: { results: BenchmarkRow[]; combined_normalized: number }
  error?: string
}> {
  const r = await apiFetch(`/api/evaluate/status/${id}`)
  return r.json()
}

/**
 * Bust the server-side result cache for a specific cache key.
 * Returns the number of DB rows deleted (0 = key was already absent).
 */
export async function invalidateCache(cacheKey: string): Promise<void> {
  await apiFetch(`/api/cache/${encodeURIComponent(cacheKey)}`, { method: 'DELETE' })
}

/**
 * Fetch combined job-pool + store stats.
 */
export async function fetchStats(): Promise<Record<string, unknown>> {
  const r = await apiFetch('/api/stats')
  return r.json()
}

// ---------------------------------------------------------------------------
// C2 dossier (R4 — additive, judge-facing)
// ---------------------------------------------------------------------------

export async function fetchDossier(runId: string): Promise<Record<string, unknown>> {
  const r = await apiFetch(`/api/reports/${runId}/dossier`)
  return r.json()
}

// ---------------------------------------------------------------------------
// C7 demo fixtures (additive — manifest written by scripts/warm_demo.py)
// ---------------------------------------------------------------------------

export interface FixtureEntry {
  id: string
  title: string
  task: string
  sampleNames: string[]
  query: string
  expected_task: string
  status: string      // 'green' | 'degraded' | 'fail'
  cached?: boolean
  run_id?: string
  error?: string
}

export interface FixturesManifest {
  fixtures: FixtureEntry[]
  all_green: boolean
  generated_by: string
}

export async function fetchFixtures(): Promise<FixturesManifest | null> {
  try {
    const r = await apiFetch('/api/fixtures')
    return r.json()
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) return null
    throw err
  }
}
