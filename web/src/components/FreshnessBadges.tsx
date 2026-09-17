// C3 — FreshnessBadges: renders the `freshness` clocks emitted on every
// enriched run. Pure renderer; absent key -> renders nothing.

export interface FreshnessData {
  generated_at?: string
  latest_obs?: string | null
  quality?: 'ok' | 'degraded' | string
  staleness_days?: number | null
  threshold_days?: number
  clocks?: Record<string, string>
  method_note?: string
  why?: string | null
}

function fmtDate(iso?: string | null): string {
  if (!iso || iso === 'unknown') return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return String(iso).slice(0, 10)
  return d.toISOString().slice(0, 10)
}

export default function FreshnessBadges({ freshness }: { freshness?: FreshnessData | null }) {
  if (!freshness) return null
  const ok = freshness.quality === 'ok'
  return (
    <span className="group relative inline-flex">
      <span className={`inline-flex items-center gap-1.5 rounded border px-2 py-0.5 font-mono text-xs ${
        ok ? 'border-good/40 bg-good/10 text-good'
           : 'border-warn/50 bg-warn-soft text-warn'}`}>
        <span className={`h-1.5 w-1.5 rounded-full ${ok ? 'bg-good' : 'bg-warn'}`} />
        {ok ? 'fresh' : 'degraded'}
        {typeof freshness.staleness_days === 'number' && (
          <span className="text-muted">{freshness.staleness_days}d / {freshness.threshold_days}d</span>
        )}
      </span>
      <span className="pointer-events-none absolute right-0 top-full z-30 mt-2 w-80 rounded-lg border border-line bg-panel p-3 text-left opacity-0 shadow-xl transition-opacity duration-200 group-hover:opacity-100">
        <div className="font-mono text-xs leading-relaxed text-muted">
          <div className="text-body">latest obs {fmtDate(freshness.latest_obs)}</div>
          <div>staleness {freshness.staleness_days ?? '—'}d (threshold {freshness.threshold_days}d)</div>
          <div className="mt-1 text-xs">{freshness.method_note}</div>
          {freshness.why && <div className="mt-1 text-warn">why: {freshness.why}</div>}
        </div>
      </span>
    </span>
  )
}