import { useEffect, useRef, useState } from 'react'
import {
  createJob, fetchGeo, fetchProvenance, pollJob,
  type GeoJSON, type JobResult,
} from '../api'
import { taskLabel } from '../labels'
import { Panel, Term } from './Console'
import MapView from './MapView'
import FreshnessBadges from './FreshnessBadges'
import HonestyBanner from './HonestyBanner'
import FixtureChip from './FixtureChip'

function ConfidenceRing({ value }: { value: number }) {
  const r = 30, C = 2 * Math.PI * r
  const off = C * (1 - value)
  const col = value > .66 ? 'var(--c-good)' : value > .4 ? 'var(--c-warn)' : 'var(--c-bad)'
  return (
    <div className="group relative h-[80px] w-[80px] shrink-0">
      <svg viewBox="0 0 76 76" className="h-full w-full -rotate-90">
        <circle cx="38" cy="38" r={r} fill="none" stroke="var(--c-line)" strokeWidth="6" />
        <circle cx="38" cy="38" r={r} fill="none" stroke={col} strokeWidth="6"
          strokeLinecap="round" strokeDasharray={C} strokeDashoffset={off}
          style={{ transition: 'stroke-dashoffset 1s ease' }} />
      </svg>
      <span className="absolute inset-0 flex items-center justify-center font-mono text-[16.5px] font-medium text-body">
        {Math.round(value * 100)}%
      </span>
      <div className="pointer-events-none absolute left-0 top-full z-30 mt-2 w-72 rounded-lg border border-line bg-panel p-3 opacity-0 shadow-xl transition-opacity duration-200 group-hover:opacity-100">
        <div className="text-[13.5px] font-medium leading-snug text-body">
          How sure the specialist is, tuned so 80% has historically meant about
          80% correct.
        </div>
        <div className="mt-2 font-mono text-[11.5px] leading-relaxed text-accent">
          confidence = ½ base + signal + specificity − assumed-GSD penalty
        </div>
      </div>
    </div>
  )
}

// OUTPUT-SECTIONS v1 — injected into Results.tsx by _apply.py (G-postfix).
// Composer + per-key section renderers. Each section returns null when its
// inputs are absent, so no backend key can silently disappear.

function StructuredOutputs({ result }: { result: JobResult }) {
  const o = (result.outputs ?? {}) as Record<string, any>
  const task = result.selected_task

  const impact = task === 'investigation'
    ? (o.investigation?.impact ?? null)
    : (task === 'impact_analysis' ? o : null)

  return (
    <div className="space-y-4">
      {impact && impact.findings && <ImpactSection impact={impact} />}
      {o.fused_classes && <FusionSection o={o} runId={result.run_id} />}
      {o.changed_area_fraction != null && <ChangeSection o={o} />}
      {(o.concept || (o.boxes?.length ?? 0) > 0 || o.ranked) && <GroundingSection o={o} result={result} />}
      {Array.isArray(o.labels) && <LabelChips o={o} />}
    </div>
  )
}

function ImpactSection({ impact }: { impact: any }) {
  if (!impact?.findings) return null
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Changed area" value={`${impact.changed_area_ha} ha`} />
        <Stat label="Changed pixels" value={`${(impact.changed_fraction * 100).toFixed(1)}%`} />
        <Stat label="Within 500 m of water" value={`${((impact.near_water?.within_500m ?? 0) * 100).toFixed(0)}%`} />
        <Stat label="Priority zone" value={impact.priority_zone || '—'} />
      </div>
      <div className="overflow-hidden rounded-lg border border-line">
        <table className="w-full text-left text-[14.5px]">
          <thead>
            <tr className="border-b border-line bg-elev text-[13px] uppercase tracking-wider text-muted">
              <th className="px-3 py-2 font-medium">Finding</th>
              <th className="px-3 py-2 font-medium">Where</th>
              <th className="px-3 py-2 font-medium">Recommended action</th>
            </tr>
          </thead>
          <tbody>
            {impact.findings.map((f: any, i1: number) => (
              <tr key={i1} className="border-b border-line/60 last:border-0">
                <td className="px-3 py-2.5 font-medium text-body">{f.what}</td>
                <td className="px-3 py-2.5 text-muted">{f.where}</td>
                <td className="px-3 py-2.5 text-body">{f.action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {impact.zones_top?.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {impact.zones_top.map((z: any) => (
            <span key={z.zone} className="rounded-lg border border-line bg-elev px-2.5 py-1 font-mono text-[13.5px] text-body">
              {z.zone} · {z.area_ha} ha · water {Math.round(z.near_water_frac * 100)}%
            </span>
          ))}
        </div>
      )}
      {impact.gsd_assumed && (
        <p className="text-[14px] italic text-warn">
          Ground resolution assumed 10 m/pixel (input is not georeferenced): areas are estimates.
        </p>
      )}
    </div>
  )
}
function FusionSection({ o, runId }: { o: Record<string, any>; runId: string }) {
  if (!o.fused_classes) return null
  const entries = Object.entries(o.fused_classes as Record<string, number>).slice(0, 6)
  const am = o.agreement_map as Record<string, any> | undefined
  const fractions = (am?.fractions ?? {}) as Record<string, number>
  return (
    <div className="space-y-4">
      <div className="space-y-1.5">
        {entries.map(([k, v]) => (
          <div key={k} className="flex items-center gap-3">
            <span className="w-56 shrink-0 truncate text-[15px] text-body">{k}</span>
            <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-elev">
              <div className="h-full rounded-full bg-accent transition-all duration-700"
                style={{ width: `${(v as number) * 100}%` }} />
            </div>
            <span className="w-12 text-right font-mono text-xs text-muted">{(v as number).toFixed(2)}</span>
          </div>
        ))}
      </div>
      {o.agreement && (
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(o.agreement as Record<string, string>).map(([k, v]) => (
            <span key={k} className="rounded-full border border-line bg-elev px-2.5 py-0.5 font-mono text-[13px] uppercase tracking-wide text-muted">
              {k}: {v}
            </span>
          ))}
        </div>
      )}
      {am && Object.keys(fractions).length > 0 && (
        <figure className="overflow-hidden rounded-lg border border-line">
          {am.overlay_png && (
            <img src={`/api/reports/${runId}/agreement_overlay.png`} alt="per-pixel optical-SAR agreement"
              className="w-full" />
          )}
          <figcaption className="border-t border-line bg-elev px-3 py-2">
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(fractions).map(([k, v]) => (
                <span key={k} className="rounded-full border border-line bg-panel px-2.5 py-0.5 font-mono text-[12.5px] text-body">
                  {k}: {(Number(v) * 100).toFixed(1)}%
                </span>
              ))}
            </div>
            {Array.isArray(am.notes) && am.notes.length > 0 && (
              <p className="mt-1.5 text-[13.5px] italic text-muted">
                {(am.notes as string[]).join(' ')}
              </p>
            )}
            {am.geotiff && (
              <a href={`/api/reports/${runId}/agreement.tif`} download
                className="mt-1.5 inline-block rounded-lg border border-line bg-panel px-3 py-1.5 text-xs text-muted transition-colors hover:border-accent/50 hover:text-accent">
                ⬇ agreement map (.tif)
              </a>
            )}
          </figcaption>
        </figure>
      )}
    </div>
  )
}

function ChangeSection({ o }: { o: Record<string, any> }) {
  if (o.changed_area_fraction == null) return null
  const transitions = (o.transitions?.top ?? []) as {
    from_class?: string; to_class?: string; from?: string; to?: string;
    area_ha?: number; pct?: number; count?: number
  }[]
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        {(o.increased as string[] | undefined)?.map((c) => (
          <span key={c} className="rounded-md border border-good/40 bg-good/10 px-2.5 py-1 text-[14.5px] text-good">▲ {c}</span>
        ))}
        {(o.decreased as string[] | undefined)?.map((c) => (
          <span key={c} className="rounded-md border border-bad/40 bg-bad/10 px-2.5 py-1 text-[14.5px] text-bad">▼ {c}</span>
        ))}
        <span className="rounded-md border border-line bg-elev px-2.5 py-1 font-mono text-xs text-body">
          changed {(Number(o.changed_area_fraction) * 100).toFixed(1)}%
        </span>
        {o.dominant_direction ? (
          <span className="rounded-md border border-line bg-elev px-2.5 py-1 font-mono text-xs text-body">
            region: {String(o.dominant_direction)}
          </span>
        ) : null}
      </div>
      {transitions.length > 0 && (
        <div className="overflow-hidden rounded-lg border border-line">
          <table className="w-full text-left text-[14.5px]">
            <thead>
              <tr className="border-b border-line bg-elev text-[13px] uppercase tracking-wider text-muted">
                <th className="px-3 py-2 font-medium">From</th>
                <th className="px-3 py-2 font-medium">To</th>
                <th className="px-3 py-2 font-medium">Area / share</th>
              </tr>
            </thead>
            <tbody>
              {transitions.slice(0, 5).map((t, i) => (
                <tr key={i} className="border-b border-line/60 last:border-0">
                  <td className="px-3 py-2.5 text-body">{t.from_class ?? t.from ?? '—'}</td>
                  <td className="px-3 py-2.5 text-body">{t.to_class ?? t.to ?? '—'}</td>
                  <td className="px-3 py-2.5 font-mono text-[13px] text-muted">
                    {typeof t.area_ha === 'number' ? `${t.area_ha} ha` : ''}
                    {typeof t.pct === 'number' ? ` ${t.pct.toFixed(1)}%` : ''}
                    {typeof t.count === 'number' ? ` · ${t.count} px` : ''}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {o.transition_summary && (
        <p className="text-[13.5px] italic text-muted">{String(o.transition_summary)}</p>
      )}
    </div>
  )
}
function GroundingSection({ o, result }: { o: Record<string, any>; result: JobResult }) {
  const boxes = (o.boxes ?? []) as number[][]
  const concept = o.concept as string | undefined
  const ranked = o.ranked as { primary?: any; alternates?: any[]; why?: string } | undefined
  if (!concept && boxes.length === 0 && !ranked) return null
  const topBox = (boxes[0] ?? ranked?.primary?.box) as number[] | undefined
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Concept" value={String(concept ?? '—')} />
        <Stat label="Regions" value={String(o.num_regions ?? boxes.length)} />
        <Stat label="Area share"
          value={typeof o.area_fraction === 'number' ? `${(o.area_fraction * 100).toFixed(1)}%` : '—'} />
        <Stat label="Verdict" value={String(o.verified ? 'verified' : (o.pending ? 'uncertain' : '—'))} />
      </div>
      {ranked?.primary && (
        <div className="rounded-lg border border-good/40 bg-good/10 px-3 py-2">
          <div className="text-[13px] font-medium uppercase tracking-wider text-good">Primary region</div>
          <div className="mt-0.5 font-mono text-[13px] text-body">
            {JSON.stringify(ranked.primary.box ?? ranked.primary)}
          </div>
          {ranked.primary.why && (
            <p className="mt-0.5 text-[13.5px] italic text-muted">{String(ranked.primary.why)}</p>
          )}
        </div>
      )}
      {Array.isArray(ranked?.alternates) && ranked.alternates.length > 0 && (
        <div className="space-y-1.5">
          <div className="text-[13px] uppercase tracking-wider text-faint">Alternates</div>
          {ranked.alternates.slice(0, 2).map((a: any, i: number) => (
            <div key={i} className="rounded-lg border border-line bg-elev px-3 py-2">
              <div className="font-mono text-[13px] text-body">
                {JSON.stringify(a.box ?? a)}
              </div>
              {a.why && (
                <p className="mt-0.5 text-[13.5px] italic text-muted">{String(a.why)}</p>
              )}
            </div>
          ))}
        </div>
      )}
      {boxes.length > 0 && (
        <p className="text-[13.5px] text-muted">
          {boxes.length} bounding {boxes.length === 1 ? 'box' : 'boxes'} — overlaid on the map
          when the input is georeferenced; drag a rectangle below to ask about any
          single region.
        </p>
      )}
      {(result.inputs?.length ?? 0) === 1 && topBox && (
        <div className="mt-2 rounded-lg border border-line p-4">
          <div className="mb-2 text-[13px] uppercase tracking-wider text-faint">
            Ask about the top region
          </div>
          <RegionQuery preset={topBox} result={result} />
        </div>
      )}
    </div>
  )
}

function LabelChips({ o }: { o: Record<string, any> }) {
  if (!Array.isArray(o.labels)) return null
  return (
    <div className="flex flex-wrap gap-1.5">
      {(o.labels as [string, number][]).map(([n, s]) => (
        <span key={n} className="rounded-full border border-line bg-elev px-2.5 py-1 text-[14.5px] text-body">
          {n} <span className="font-mono text-[13px] text-muted">{s}</span>
        </span>
      ))}
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-line bg-elev px-3 py-2.5">
      <div className="text-[13px] uppercase tracking-wider text-faint">{label}</div>
      <div className="mt-0.5 font-mono text-[16.5px] font-medium text-body">{value}</div>
    </div>
  )
}

export default function Results({ result, onFollowUp, onRequestSwitch }:
  { result: JobResult; onFollowUp?: (q: string) => void;
    onRequestSwitch?: (m: 'sar' | 'optical') => void }) {
  const [tab, setTab] = useState<'evidence' | 'compare' | 'map' | 'json'>('evidence')
  const [prov, setProv] = useState<any>(null)
  const hasCompare = (result.inputs?.length ?? 0) === 2
  const hasGeo = !!result.inputs?.[0]?.summary?.georeferenced
  const evidenceImgs = Object.entries(result.visual_data ?? {})
  const outputs = (result.outputs ?? {}) as Record<string, any>
  const freshness = outputs.freshness as {
    generated_at?: string; latest_obs?: string | null; quality?: string;
    staleness_days?: number | null; threshold_days?: number;
    clocks?: Record<string, string>; method_note?: string; why?: string | null
  } | undefined
  const honesty = outputs.honesty as {
    fallback_active?: boolean; below_gate?: unknown[]; pixel_space?: boolean;
    limitation_refs?: string[]
  } | undefined

  useEffect(() => { fetchProvenance().then(setProv).catch(() => {}) }, [])

  const modelChip = (() => {
    if (!prov) return null
    const map: Record<string, string> = {
      single_vqa: 'VQA Specialist', captioning: 'Scene Encoder',
      grounding: 'Scene Encoder', change_analysis: 'Change Detector',
      change_vqa: 'Change Detector', impact_analysis: 'Change Detector',
      investigation: 'Change Detector', optical_sar: 'Optical-SAR Fusion',
    }
    const want = map[result.selected_task]
    const card = prov.models?.find((m: any) => m.component === want)
    if (!card) return null
    return (
      <span className="rounded border border-line bg-elev px-2 py-0.5 font-mono text-[13px] text-muted">
        {card.component}{typeof card.val_accuracy === 'number'
          ? ` · val ${(card.val_accuracy * 100).toFixed(0)}%` : ''}
      </span>
    )
  })()

  const suggestions: string[] = (result.outputs?.suggestions as string[]) ?? []

  return (
    <div className="fade-up space-y-6">
      {/* answer card */}
      <div className="relative overflow-hidden rounded-xl border border-line bg-panel p-6 shadow-[var(--shadow-panel)]">
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-accent/60 to-transparent" />
        <div className="flex items-start gap-6">
          <ConfidenceRing value={result.confidence} />
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded border border-accent/40 bg-accent-soft px-2 py-0.5 text-[14px] font-semibold uppercase tracking-wide text-accent">
                {taskLabel(result.selected_task)}
              </span>
              {modelChip}
              {result.cached && (
                <span className="rounded border border-line bg-elev px-2 py-0.5 font-mono text-[13px] text-muted">cached</span>
              )}
              <FreshnessBadges freshness={freshness} />
              <FixtureChip result={result} />
            </div>
            <p className="mt-3 text-[18px] leading-relaxed text-body">{result.answer}</p>
            <div className="mt-3">
              <HonestyBanner honesty={honesty} />
            </div>
          </div>
        </div>
        <div className="mt-5 flex flex-wrap items-center gap-2">
          {result.run_id && (
            <a href={`/api/reports/${result.run_id}/report.pdf`} download
              className="rounded-lg border border-line px-3 py-1.5 text-xs text-muted transition-colors hover:border-accent/50 hover:text-accent">
              ⬇ report.pdf
            </a>
          )}
          {result.run_id && (
            <>
              <a href={`/api/reports/${result.run_id}/report.md`} download
                className="rounded-lg border border-line px-3 py-1.5 text-xs text-muted transition-colors hover:border-accent/50 hover:text-accent">
                ⬇ report.md
              </a>
              <a href={`/api/reports/${result.run_id}/report.json`} download
                className="rounded-lg border border-line px-3 py-1.5 text-xs text-muted transition-colors hover:border-accent/50 hover:text-accent">
                ⬇ report.json
              </a>
              {['change_analysis', 'investigation', 'impact_analysis'].includes(result.selected_task) && (
                <a href={`/api/reports/${result.run_id}/change_mask.tif`} download
                  className="rounded-lg border border-line px-3 py-1.5 text-xs text-muted transition-colors hover:border-accent/50 hover:text-accent">
                  ⬇ change mask (.tif)
                </a>
              )}
            </>
          )}

        </div>
      </div>

      {suggestions.length > 0 && onFollowUp && (
        <div>
          <Term t="Ask the data back" d="One click launches a follow-up analysis on the same imagery: the investigative loop." />
          <div className="mt-2 flex flex-wrap gap-2">
            {suggestions.map(q => (
              <button key={q} onClick={() => onFollowUp(q)}
                className="rounded-full border border-accent/40 bg-accent-soft px-3.5 py-1.5 text-[14.5px] text-accent transition-colors hover:bg-accent/20">
                {q} →
              </button>
            ))}
          </div>
        </div>
      )}

      <StructuredOutputs result={result} />

      {/* evidence */}
      <Panel title="Visual evidence" hint="Overlays highlight exactly which pixels support the answer.">
        <div className="mb-4 flex gap-1 rounded-lg bg-elev p-1">
          {([['evidence', 'Evidence'], ['compare', 'Swipe compare'],
             ['map', 'Map'], ['json', 'Outputs']] as const).map(([k, l]) => (
            <button key={k} onClick={() => setTab(k)}
              disabled={(k === 'compare' && !hasCompare) || (k === 'map' && !hasGeo)}
              className={`flex-1 rounded-md px-3 py-1.5 text-[14.5px] transition-colors disabled:opacity-35 ${
                tab === k ? 'bg-panel text-body shadow-sm' : 'text-muted hover:text-body'}`}>
              {l}
            </button>
          ))}
        </div>

        {tab === 'evidence' && (
          <>
            {evidenceImgs.length ? (
              <div className={`grid gap-4 ${evidenceImgs.length > 1 ? 'md:grid-cols-2' : ''}`}>
                {evidenceImgs.map(([k, url]) => (
                  <figure key={k} className="overflow-hidden rounded-lg border border-line">
                    {url && <img src={url} alt={k} className="w-full" />}
                    <figcaption className="border-t border-line bg-elev px-3 py-2 font-mono text-[13px] text-muted">
                      {k.split('_').join(' ')}
                    </figcaption>
                  </figure>
                ))}
              </div>
            ) : (
              <InputsGrid result={result} onRequestSwitch={onRequestSwitch} />
            )}
            {(result.inputs?.length ?? 0) === 1 && (
              <div className="mt-4">
                <RegionQuery result={result} />
              </div>
            )}
          </>
        )}

        {tab === 'compare' && hasCompare && <SwipeCompare result={result} />}
        {tab === 'map' && hasGeo && <MapWithGeo runId={result.run_id} />}

        {tab === 'json' && (
          <div className="min-w-0 overflow-hidden rounded-lg border border-line bg-elev">
            <pre className="max-h-[420px] max-w-full overflow-auto whitespace-pre-wrap break-words p-4 font-mono text-[13px] leading-relaxed text-muted">
              {JSON.stringify(result.outputs, null, 2)}
            </pre>
          </div>
        )}
      </Panel>

      <Panel title="Why trust this answer?">
        <ul className="space-y-1.5 text-[15px] text-muted">
          <li>✓ Every step recorded: validation, routing, tool selection, execution timings (see the execution trace above).</li>
          <li>✓ Models fine-tuned on public remote-sensing benchmarks: provenance and metrics in the Provenance tab.</li>
          <li>✓ Confidence is temperature-calibrated against held-out data, not a raw softmax.</li>
          <li>✓ Visual evidence is derived from the same pixels the models saw: click the overlays to verify.</li>
        </ul>
      </Panel>
    </div>
  )
}

function MapWithGeo({ runId }: { runId: string }) {
  const [geo, setGeo] = useState<GeoJSON | null>(null)
  const [err, setErr] = useState('')
  useEffect(() => {
    fetchGeo(runId).then(setGeo).catch(e => setErr(String(e)))
  }, [runId])
  if (err) return <p className="text-sm text-muted">Map unavailable: {err}</p>
  if (!geo) return <p className="text-sm text-muted">Loading overlay…</p>
  return <MapView geo={geo} />
}

function ModalityBadge({ summary, onRequestSwitch }:
  { summary: Record<string, any>; onRequestSwitch?: (m: 'sar' | 'optical') => void }) {
  const mc = summary.modality_certainty as {
    label?: string; confidence?: number; votes?: Record<string, unknown>
  } | null | undefined
  if (!mc?.label) return null
  const votes = mc.votes ?? {}
  const conflict = typeof votes.filename === 'string'
    && typeof votes.stats === 'string'
    && votes.filename !== votes.stats
  return (
    <span className="group relative inline-flex">
      <span className="inline-flex items-center gap-1 rounded border border-line bg-panel px-1.5 py-0.5 font-mono text-[12.5px] uppercase text-body">
        {String(mc.label)}
        {typeof mc.confidence === 'number' && (
          <span className="text-muted">{Math.round(mc.confidence * 100)}%</span>
        )}
      </span>
      <span className="pointer-events-none absolute bottom-full left-1/2 z-30 mb-2 w-72 -translate-x-1/2 rounded-lg border border-line bg-panel p-3 text-left opacity-0 shadow-xl transition-opacity duration-200 group-hover:opacity-100">
        <div className="font-mono text-[12.5px] leading-relaxed text-muted">
          <div>filename vote: {String(votes.filename ?? '—')}</div>
          <div>stats vote: {String(votes.stats ?? '—')}</div>
          {votes.override !== undefined && (
            <div>override: {String(votes.override)}</div>
          )}
          <div className="mt-1 text-[11.5px]">
            Stats win over filename (median, pixel sign, band count). Re-run with an
            explicit override if this looks wrong.
          </div>
        </div>
      </span>
      {conflict && onRequestSwitch && (
        <button onClick={() => onRequestSwitch('sar')}
          className="ml-1 rounded border border-warn/50 bg-warn-soft px-1.5 py-0.5 font-mono text-[12.5px] text-warn hover:bg-warn/20">
          looks like SAR — switch?
        </button>
      )}
    </span>
  )
}

function InputsGrid({ result, onRequestSwitch }: {
  result: JobResult
  onRequestSwitch?: (m: 'sar' | 'optical') => void
}) {
  return (
    <div className={`grid gap-4 ${(result.inputs?.length ?? 0) > 1 ? 'md:grid-cols-2' : ''}`}>
      {result.inputs?.map((inp, i) => (
        <figure key={i} className="overflow-hidden rounded-lg border border-line">
          <img src={inp.composite} alt={inp.summary.file} className="w-full bg-black/5" />
          <figcaption className="flex flex-wrap items-center gap-2 border-t border-line bg-elev px-3 py-2">
            <span className="truncate font-mono text-[13px] text-muted">{inp.summary.file}</span>
            {inp.summary.acquired && (
              <span className="rounded border border-line bg-panel px-1.5 py-0.5 font-mono text-[12.5px] text-body">
                📅 {String(inp.summary.acquired)}
              </span>
            )}
            <span className="ml-auto flex items-center gap-2">
              <ModalityBadge summary={inp.summary as unknown as Record<string, any>}
                onRequestSwitch={onRequestSwitch} />
              <span className="font-mono text-[13px] uppercase text-faint">{inp.summary.modality}</span>
            </span>
          </figcaption>
        </figure>
      ))}
    </div>
  )
}

function SwipeCompare({ result }: { result: JobResult }) {
  const [pos, setPos] = useState(50)
  const ref = useRef<HTMLDivElement>(null)
  const imgs = result.inputs ?? []
  if (imgs.length < 2) return null

  function move(clientX: number) {
    const rect = ref.current?.getBoundingClientRect()
    if (!rect) return
    setPos(Math.min(96, Math.max(4, ((clientX - rect.left) / rect.width) * 100)))
  }

  return (
    <div>
      <div ref={ref}
        onMouseMove={(e) => e.buttons === 1 && move(e.clientX)}
        onTouchMove={(e) => move(e.touches[0].clientX)}
        className="relative select-none overflow-hidden rounded-lg border border-line">
        <img src={imgs[1].composite} alt="B" className="block w-full" draggable={false} />
        <img src={imgs[0].composite} alt="A" draggable={false}
          className="absolute inset-0 block h-full w-full object-cover"
          style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }} />
        <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-accent shadow-[0_0_10px_var(--c-accent)]"
          style={{ left: `${pos}%` }}>
          <span className="absolute top-1/2 -ml-4 h-8 w-8 -translate-y-1/2 rounded-full border-2 border-accent bg-panel" />
        </div>
        <span className="absolute left-3 top-3 rounded bg-black/55 px-2 py-0.5 font-mono text-[13px] uppercase tracking-wide text-white">A · before</span>
        <span className="absolute right-3 top-3 rounded bg-black/55 px-2 py-0.5 font-mono text-[13px] uppercase tracking-wide text-white">B · after</span>
      </div>
      <input type="range" min={4} max={96} value={pos} onChange={(e) => setPos(+e.target.value)}
        className="mt-3 w-full accent-[var(--c-accent)]" />
    </div>
  )
}

// Shared prompt for region interrogation — single source of truth.
export const REGION_QUERY_PROMPT = 'Describe the land-cover and major objects visible in this image.'

function RegionQuery({ result, preset }: { result: JobResult; preset?: number[] }) {
  const [rect, setRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null)
  const [busy, setBusy] = useState(false)
  const [answer, setAnswer] = useState<string>('')
  const imgRef = useRef<HTMLImageElement>(null)
  const startRef = useRef<{ x: number; y: number } | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  // G7: per-box "Ask about this region" — when a grounding box preset is
  // passed, pre-fill the drag rectangle once the image is laid out.
  const presetApplied = useRef(false)

  // Cancel any in-flight poll on unmount to avoid setState-after-unmount leaks
  // and to stop server-side jobs that the user can no longer see.
  useEffect(() => () => { abortRef.current?.abort() }, [])

  const inp = result.inputs?.[0]
  if (!inp) return null

  function onDown(e: React.MouseEvent) {
    const r = (e.target as HTMLElement).getBoundingClientRect()
    startRef.current = { x: e.clientX - r.left, y: e.clientY - r.top }
    setRect(null)
  }
  function onMove(e: React.MouseEvent) {
    if (!startRef.current || e.buttons !== 1) return
    const r = (e.target as HTMLElement).getBoundingClientRect()
    const cx = e.clientX - r.left, cy = e.clientY - r.top
    const s = startRef.current
    setRect({ x: Math.min(s.x, cx), y: Math.min(s.y, cy),
              w: Math.abs(cx - s.x), h: Math.abs(cy - s.y) })
  }
  async function ask() {
    const img = imgRef.current
    if (!img || !rect) return
    const sx = img.naturalWidth / img.clientWidth
    const sy = img.naturalHeight / img.clientHeight
    // Clamp the source rectangle to the image dimensions so drawImage
    // never receives out-of-bounds source coords (which would throw).
    const srcX = Math.max(0, rect.x * sx)
    const srcY = Math.max(0, rect.y * sy)
    const srcW = Math.min(img.naturalWidth - srcX, rect.w * sx)
    const srcH = Math.min(img.naturalHeight - srcY, rect.h * sy)
    const canvas = document.createElement('canvas')
    canvas.width = Math.max(8, Math.round(srcW))
    canvas.height = Math.max(8, Math.round(srcH))
    const ctx = canvas.getContext('2d')
    if (!ctx || canvas.width === 0 || canvas.height === 0) {
      setAnswer('Could not read the selected region — try again.')
      return
    }
    ctx.drawImage(img, srcX, srcY, srcW, srcH, 0, 0, canvas.width, canvas.height)
    const blob = await new Promise<Blob | null>(res =>
      canvas.toBlob(res, 'image/png'))
    // canvas.toBlob can return null (Safari / low memory) — handle gracefully.
    if (!blob) {
      setAnswer('Could not encode the selected region — try a smaller area or another browser.')
      return
    }
    // Cancel any previous in-flight poll before starting a new one.
    abortRef.current?.abort()
    abortRef.current = new AbortController()
    setBusy(true); setAnswer('')
    try {
      const jid = await createJob({
        query: REGION_QUERY_PROMPT,
        files: [new File([blob], 'region.png', { type: 'image/png' })],
        sampleNames: [],
      })
      const MAX_POLLS = 240
      for (let i = 0; i < MAX_POLLS; i++) {
        const st = await pollJob(jid, abortRef.current?.signal)
        if (st.status === 'done') { setAnswer(st.result?.answer ?? ''); break }
        if (st.status === 'error') { setAnswer('failed: ' + (st.error ?? '')); break }
        if (i === MAX_POLLS - 1) {
          setAnswer('Region analysis timed out — try a smaller region.')
        }
        await new Promise(r => setTimeout(r, 400))
      }
    } catch (e) {
      if (!abortRef.current?.signal.aborted) {
        setAnswer('Region analysis failed — try again.')
      }
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="rounded-lg border border-line p-4">
      <div className="mb-2 text-[13px] uppercase tracking-wider text-faint">
        Interrogate a region: drag a rectangle, then ask
      </div>
      <div className="relative inline-block max-w-full overflow-hidden rounded border border-line"
        onMouseDown={onDown} onMouseMove={onMove} onMouseUp={() => startRef.current = null}>
        <img ref={imgRef} src={inp.composite} alt="region select" className="block max-w-full select-none" draggable={false}
          onLoad={() => {
            if (presetApplied.current || !preset || preset.length < 4) return
            presetApplied.current = true
            const img = imgRef.current
            if (!img) return
            // Normalised 0..1 box (as emitted by grounding ranked output) ->
            // display pixels of the scaled <img>.
            const [x1, y1, x2, y2] = preset
            const w = img.clientWidth, h = img.clientHeight
            if (w > 0 && h > 0) {
              setRect({
                x: Math.min(x1, x2) * w, y: Math.min(y1, y2) * h,
                w: Math.abs(x2 - x1) * w, h: Math.abs(y2 - y1) * h,
              })
            }
          }} />
        {rect && (
          <div className="pointer-events-none absolute border-2 border-accent bg-accent-soft"
            style={{ left: rect.x, top: rect.y, width: rect.w, height: rect.h }} />
        )}
      </div>
      <button onClick={ask} disabled={!rect || busy}
        className="mt-3 block rounded-lg bg-accent px-4 py-1.5 text-xs font-semibold text-white hover:bg-accent-dim disabled:opacity-40">
        {busy ? 'Analysing region…' : 'Ask about this region'}
      </button>
      {answer && (
        <p className="mt-3 rounded-lg border border-line bg-elev px-3 py-2 text-sm text-body">{answer}</p>
      )}
    </div>
  )
}
