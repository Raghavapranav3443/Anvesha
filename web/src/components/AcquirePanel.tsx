/**
 * Online acquisition panel: "tell me a place, I will find the imagery".
 *
 * This is the user-facing half of the online layer, written for someone who has
 * no GeoTIFF, does not know what a scene ID is, and cannot read coordinates. It
 * is deliberately two steps rather than one:
 *
 *   1. **Find imagery** resolves the place and lists the satellite passes it is
 *      willing to use, with dates and cloud cover. Nothing is downloaded.
 *   2. **Fetch & analyse** downloads the chosen pair and hands it to the normal
 *      pipeline.
 *
 * Showing step 1 is not decoration. A fetch takes about a minute, the choice of
 * two dates decides whether change is detectable at all, and someone about to
 * spend that minute deserves to see which passes were picked.
 *
 * Failures are phrased as actions: air-gap mode offers the switch, an unmatched
 * place suggests the accepted format, and an empty catalogue suggests a longer
 * date range instead of reporting "no data".
 */
import { useEffect, useState } from 'react'
import {
  ApiError,
  type AcquireOutcome,
  type AcquirePlan,
  type AcquireStatus,
  fetchAcquire,
  fetchAcquireStatus,
  planAcquire,
  setMode,
} from '../api'

const PLACE_PRESETS = [
  'Dibrugarh, Assam',
  'Kamrup, Assam',
  'Patna, Bihar',
  'Kochi, Kerala',
]

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-3 py-0.5 text-xs">
      <span className="min-w-[7.5rem] shrink-0 text-faint">{label}</span>
      <span className="text-ink">{children}</span>
    </div>
  )
}

function Chip({ children, tone = 'muted' }: {
  children: React.ReactNode
  tone?: 'muted' | 'ok' | 'warn'
}) {
  const cls = tone === 'ok'
    ? 'border-emerald-500/50 text-emerald-400'
    : tone === 'warn'
      ? 'border-amber-500/50 text-amber-400'
      : 'border-line text-muted'
  return (
    <span className={`mr-1.5 inline-block whitespace-nowrap rounded border px-1.5 py-px font-mono text-[10px] uppercase tracking-wider ${cls}`}>
      {children}
    </span>
  )
}

export default function AcquirePanel({
  onAcquired, busy = false,
}: {
  onAcquired: (acquireId: string, outcome: AcquireOutcome) => void
  busy?: boolean
}) {
  const [status, setStatus] = useState<AcquireStatus | null>(null)
  const [place, setPlace] = useState('')
  const [days, setDays] = useState(240)
  const [maxCloud, setMaxCloud] = useState(20)
  const [plan, setPlan] = useState<AcquirePlan | null>(null)
  const [outcome, setOutcome] = useState<AcquireOutcome | null>(null)
  const [phase, setPhase] = useState<'idle' | 'planning' | 'fetching'>('idle')
  const [error, setError] = useState('')

  async function loadStatus() {
    try {
      const s = await fetchAcquireStatus()
      setStatus(s)
      if (s.defaults?.days) setDays(s.defaults.days)
      if (s.defaults?.max_cloud) setMaxCloud(s.defaults.max_cloud)
    } catch {
      setStatus(null)
    }
  }

  useEffect(() => { loadStatus() }, [])

  const online = status?.mode === 'online'
  const idle = phase === 'idle' && !busy

  async function enableOnline() {
    setError('')
    try {
      await setMode('online')
      await loadStatus()
    } catch (e) {
      setError(String(e))
    }
  }

  async function doPlan() {
    if (!place.trim()) {
      setError('Name a place first — for example "Dibrugarh, Assam".')
      return
    }
    setPhase('planning'); setError(''); setPlan(null); setOutcome(null)
    try {
      setPlan(await planAcquire({ query: place, days, maxCloudPct: maxCloud }))
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    } finally {
      setPhase('idle')
    }
  }

  async function doFetch() {
    if (!place.trim()) { setError('Name a place first.'); return }
    setPhase('fetching'); setError('')
    try {
      const result = await fetchAcquire({ query: place, days, maxCloudPct: maxCloud })
      setOutcome(result)
      onAcquired(result.acquire_id, result)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    } finally {
      setPhase('idle')
    }
  }

  return (
    <section className="rounded-xl border border-line bg-panel p-4">
      <div className="mb-3 flex flex-wrap items-baseline gap-3">
        <h2 className="font-mono text-sm uppercase tracking-wider text-accent">
          Find imagery for me
        </h2>
        <span className="text-xs text-faint">
          Type a place; Anvesha finds the satellite passes and downloads them.
        </span>
        {status && (
          online ? <Chip tone="ok">online</Chip> : <Chip tone="warn">air-gapped</Chip>
        )}
      </div>

      {!online && status && (
        <div className="mb-3 rounded border border-amber-500/40 bg-amber-500/5 p-3">
          <p className="text-sm text-ink">
            Anvesha is in air-gapped mode, so it will not reach the network. You can
            still analyse imagery you already have.
          </p>
          <button
            onClick={enableOnline}
            className="mt-2 rounded border border-accent/60 bg-accent-soft px-3 py-1 text-xs font-medium text-accent transition-colors hover:border-accent">
            Switch to online mode
          </button>
        </div>
      )}

      <div className="mb-2 flex flex-wrap items-center gap-2">
        <input
          value={place}
          onChange={(e) => setPlace(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && online && idle) doPlan() }}
          placeholder="e.g. Dibrugarh, Assam — or coordinates like 27.48, 94.90"
          disabled={!online}
          className="min-w-[16rem] flex-1 rounded border border-line bg-elev px-3 py-2 text-sm text-ink placeholder:text-faint disabled:opacity-50"
        />
        <label className="flex items-center gap-1.5 text-xs text-faint">
          look back
          <input type="number" min={7} max={1095} value={days} disabled={!online}
            onChange={(e) => setDays(Number(e.target.value))}
            className="w-16 rounded border border-line bg-elev px-2 py-1 text-xs text-ink disabled:opacity-50" />
          days
        </label>
        <label className="flex items-center gap-1.5 text-xs text-faint">
          max cloud
          <input type="number" min={0} max={100} value={maxCloud} disabled={!online}
            onChange={(e) => setMaxCloud(Number(e.target.value))}
            className="w-14 rounded border border-line bg-elev px-2 py-1 text-xs text-ink disabled:opacity-50" />
          %
        </label>
      </div>

      <div className="mb-3 flex flex-wrap gap-1.5">
        {PLACE_PRESETS.map((p) => (
          <button key={p} onClick={() => setPlace(p)} disabled={!online}
            className="rounded-full border border-line bg-elev px-2.5 py-0.5 text-[11px] text-muted transition-colors hover:border-accent/50 hover:text-accent disabled:opacity-50">
            {p}
          </button>
        ))}
      </div>

      <div className="flex flex-wrap gap-2">
        <button onClick={doPlan} disabled={!online || !idle}
          className="rounded-lg border border-accent bg-accent-soft px-4 py-2 text-sm font-medium text-accent transition-colors hover:border-accent/70 disabled:opacity-40">
          {phase === 'planning' ? 'Searching catalogues…' : '1 · Find imagery'}
        </button>
        <button onClick={doFetch} disabled={!online || !idle}
          className="rounded-lg border border-emerald-600/60 px-4 py-2 text-sm font-medium text-emerald-400 transition-colors hover:border-emerald-500 disabled:opacity-40">
          {phase === 'fetching' ? 'Downloading imagery…' : '2 · Fetch & analyse'}
        </button>
      </div>

      {phase === 'fetching' && (
        <p className="mt-2 text-xs text-faint">
          Reading only the needed windows from the satellite files, then fetching
          ISRO's own map layers for the same area. This usually takes about a minute.
        </p>
      )}

      {error && (
        <p className="mt-3 whitespace-pre-wrap text-sm text-rose-400">{error}</p>
      )}

      {plan && (
        <div className="mt-4 border-t border-line pt-3">
          <Row label="Place">
            {plan.place.name} <span className="text-faint">({plan.place.source})</span>
          </Row>
          <Row label="Window">
            {plan.place.size_km[0]} × {plan.place.size_km[1]} km around{' '}
            {plan.place.centroid.lat.toFixed(3)}, {plan.place.centroid.lon.toFixed(3)}
          </Row>
          <Row label="Grid">
            EPSG:{plan.grid.epsg} · {plan.grid.width}×{plan.grid.height} ·{' '}
            {plan.grid.pixel_size_m} m/px · {Math.round(plan.grid.area_ha)} ha
          </Row>
          <Row label="Searched">{plan.date_range.start} → {plan.date_range.end}</Row>

          {plan.place.admin?.window_note && (
            <p className="mt-1.5 text-xs text-amber-400">{plan.place.admin.window_note}</p>
          )}

          {plan.pair.before && plan.pair.after ? (
            <div className="mt-2">
              <Row label="Will compare">
                <Chip tone="ok">
                  {plan.pair.before.date} · {plan.pair.before.cloud_pct?.toFixed(0) ?? '?'}% cloud
                </Chip>
                <span className="mr-1.5 text-faint">with</span>
                <Chip tone="ok">
                  {plan.pair.after.date} · {plan.pair.after.cloud_pct?.toFixed(0) ?? '?'}% cloud
                </Chip>
              </Row>
            </div>
          ) : (
            <p className="mt-2 text-sm text-amber-400">
              No two usable passes were found in that period. A longer date range or
              a higher cloud allowance usually helps — this area is cloudy for much
              of the year.
            </p>
          )}

          {plan.candidates.length > 0 && (
            <details className="mt-2">
              <summary className="cursor-pointer text-xs text-faint">
                {plan.scene_count} pass{plan.scene_count === 1 ? '' : 'es'} found
              </summary>
              <div className="mt-1.5 max-h-36 overflow-y-auto">
                {plan.candidates.slice(0, 20).map((c) => (
                  <div key={c.id} className="flex gap-3 py-0.5 font-mono text-[11px]">
                    <span className="w-20 text-ink">{c.date}</span>
                    <span className={`w-14 ${c.cloud_pct !== null && c.cloud_pct <= 20 ? 'text-emerald-400' : 'text-faint'}`}>
                      {c.cloud_pct === null ? 'unknown' : `${c.cloud_pct.toFixed(0)}%`}
                    </span>
                    <span className="truncate text-faint">{c.id}</span>
                  </div>
                ))}
              </div>
            </details>
          )}

          {plan.errors.length > 0 && (
            <details className="mt-2">
              <summary className="cursor-pointer text-xs text-faint">
                catalogue notes ({plan.errors.length})
              </summary>
              {plan.errors.map((e, i) => (
                <p key={i} className="py-0.5 font-mono text-[10px] text-faint">
                  {e.provider}: {e.error}
                </p>
              ))}
            </details>
          )}
        </div>
      )}

      {outcome && (
        <div className="mt-4 border-t border-line pt-3">
          <Row label="Downloaded">
            {outcome.dates.before} and {outcome.dates.after} in{' '}
            {Math.round(outcome.elapsed_s)}s
          </Row>

          {outcome.context_plain.length > 0 ? (
            <div className="mt-2">
              <p className="mb-1 text-xs text-faint">
                What ISRO's own maps say about this area
              </p>
              {outcome.context_plain.map((line, i) => (
                <p key={i} className="py-0.5 text-xs text-ink">· {line}</p>
              ))}
            </div>
          ) : (
            <p className="mt-2 text-xs text-faint">
              No ISRO thematic layer registered data for this window, so none is
              being claimed.
            </p>
          )}

          {outcome.warnings.map((w, i) => (
            <p key={i} className="py-0.5 text-xs text-amber-400">! {w}</p>
          ))}

          <p className="mt-2 text-xs text-faint">
            Analysing these two images now — the report below explains what changed,
            what it means, and what to do about it.
          </p>
        </div>
      )}

      {status && (
        <div className="mt-3 border-t border-line pt-2">
          <Row label="Imagery from">
            <span className="text-faint">{status.active_providers.join(', ')}</span>
          </Row>
          <Row label="ISRO context">
            <span className="text-faint">
              {status.isro.service} — no account required
            </span>
          </Row>
          {status.cache?.used_mb !== undefined && (
            <Row label="Local cache">
              <span className="text-faint">
                {status.cache.used_mb} MB of {status.cache.limit_mb} MB ·{' '}
                {status.cache.entries ?? 0} items
              </span>
            </Row>
          )}
        </div>
      )}
    </section>
  )
}
