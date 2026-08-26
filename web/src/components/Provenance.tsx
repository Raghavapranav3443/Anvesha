import type { Provenance } from '../api'
import { Panel } from './Console'

export default function ProvenanceView({ prov }: { prov: Provenance | null }) {
  if (!prov) return <p className="text-muted">Loading provenance…</p>
  return (
    <div className="fade-up grid grid-cols-1 gap-6 lg:grid-cols-2">
      <Panel title="Model registry — remote-sensing adaptation">
        <div className="space-y-3">
          {prov.models.map((m) => (
            <div key={m.component} className="flex items-center gap-4 rounded-lg border border-line bg-elev px-4 py-3">
              <span className={`h-2.5 w-2.5 rounded-full ${m.trained && !m.synthetic ? 'bg-emerald-500' : m.synthetic ? 'bg-amber-500' : 'bg-slate-600'}`} />
              <div className="min-w-0 flex-1">
                <div className="text-base text-body">{m.component}</div>
                <div className="truncate font-mono text-[15.5px] text-faint">{m.file}</div>
              </div>
              {typeof m.val_accuracy === 'number' && (
                <span className="font-mono text-base text-emerald-400">{(m.val_accuracy * 100).toFixed(1)}%</span>
              )}
              <span className={`rounded border px-1.5 py-px font-mono text-[14px] uppercase ${
                m.trained ? 'border-emerald-700/40 bg-emerald-900/20 text-emerald-400'
                          : 'border-line bg-slate-900 text-muted'}`}>
                {m.synthetic ? 'synthetic' : m.trained ? 'fine-tuned' : 'fallback'}
              </span>
            </div>
          ))}
        </div>
        <p className="mt-4 text-[16.5px] leading-relaxed text-muted">
          Every specialist is fine-tuned on open remote-sensing data (EuroSAT ·
          RSVQA-LR · LEVIR-CD · BigEarthNet v2 S1+S2). Weights load automatically;
          interpretable fallbacks keep the assistant usable without them.
        </p>
      </Panel>

      <Panel title="Public benchmark results (measured)">
        {prov.benchmarks?.length ? (
          <table className="w-full text-left text-[17px]">
            <thead>
              <tr className="border-b border-line text-[15.5px] uppercase tracking-wide text-faint">
                <th className="py-2 pr-2 font-medium">Benchmark</th>
                <th className="py-2 pr-2 font-medium">Metric</th>
                <th className="py-2 pr-2 font-medium">n</th>
                <th className="py-2 text-right font-medium">Score</th>
              </tr>
            </thead>
            <tbody>
              {prov.benchmarks.filter((b) => b.score != null || b.iou != null).map((b, i) => {
                const score = b.score ?? b.iou
                const extra = b.f1 != null ? ` / F1 ${b.f1}` : ''
                return (
                  <tr key={i} className="border-b border-line/60 last:border-0">
                    <td className="py-2.5 pr-2 text-body">{b.benchmark}</td>
                    <td className="py-2.5 pr-2 text-muted">{b.metric}</td>
                    <td className="py-2.5 pr-2 font-mono text-muted">{b.n}</td>
                    <td className="py-2.5 text-right font-mono text-emerald-400">
                      {score != null ? score : '—'}{extra}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        ) : (
          <p className="text-base text-muted">Run <code className="font-mono">scripts/run_benchmarks.py</code> to populate.</p>
        )}
      </Panel>
    </div>
  )
}
