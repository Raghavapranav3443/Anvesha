import { useRef, useState } from 'react'
import { evalStatus, runEvaluation, type Provenance } from '../api'
import { Panel } from './Console'

export default function EvaluationView({ prov }: { prov: Provenance | null }) {
  const [evalId, setEvalId] = useState<string | null>(null)
  const [status, setStatus] = useState<string>('')
  const [card, setCard] = useState<any | null>(null)
  const [n, setN] = useState(300)
  const idRef = useRef<string | null>(null)

  async function run() {
    const id = await runEvaluation(n)
    idRef.current = id
    setEvalId(id)
    const poll = setInterval(async () => {
      const st = await evalStatus(idRef.current!)
      setStatus(st.status)
      if (st.status === 'done') {
        setCard(st.scorecard)
        clearInterval(poll)
      } else if (st.status === 'error') {
        setStatus('error: ' + (st.error ?? '').slice(0, 160))
        clearInterval(poll)
      }
    }, 1500)
  }
  const results = card?.results ?? prov?.benchmarks ?? []
  const combined = card?.combined_normalized

  return (
    <div className="space-y-6">
      <Panel title="Benchmark & evaluation harness">
        <p className="mb-4 max-w-3xl text-base leading-relaxed text-muted">
          Reproducible evaluation over the prescribed public benchmark subsets.
          The combined score is the mean of per-benchmark normalised scores —
          the same normalisation the problem statement applies before combining.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-base text-muted">
            items per benchmark
            <input type="number" min={50} max={800} value={n}
              onChange={e => setN(+e.target.value)}
              className="w-20 rounded border border-line bg-panel px-2 py-1 font-mono text-body outline-none focus:border-accent/60" />
          </label>
          <button onClick={run} disabled={!!evalId && status === 'running'}
            className="rounded-lg bg-accent px-5 py-1.5 text-base font-semibold text-white hover:bg-accent-dim disabled:opacity-40">
            {status === 'running' ? 'Evaluating…' : 'Run evaluation'}
          </button>
          {status === 'running' && <span className="scanning relative px-2 font-mono text-base text-accent">running on server…</span>}
          {status.startsWith('error') && <span className="text-base text-red-400">{status}</span>}
        </div>
      </Panel>

      <Panel title={card ? `Scorecard — combined normalised ${combined}` : 'Latest measured scorecard'}>
        <table className="w-full text-left text-base">
          <thead>
            <tr className="border-b border-line text-sm uppercase tracking-wider text-faint">
              <th className="py-2 pr-3 font-medium">Benchmark</th>
              <th className="py-2 pr-3 font-medium">Metric</th>
              <th className="py-2 pr-3 font-medium">n</th>
              <th className="py-2 text-right font-medium">Score</th>
            </tr>
          </thead>
          <tbody>
            {results.filter((b: any) => b.score != null || b.iou != null).map((b: any, i: number) => {
              const norm = b.normalized_score ?? b.score ?? b.iou
              return (
                <tr key={i} className="border-b border-line/60">
                  <td className="py-2.5 pr-3 text-body">{b.benchmark}</td>
                  <td className="py-2.5 pr-3 text-muted">{b.metric}</td>
                  <td className="py-2.5 pr-3 font-mono text-muted">{b.n}</td>
                  <td className="py-2.5 text-right font-mono text-emerald-400">
                    {b.f1 != null ? `${b.iou} / ${b.f1}` : (b.score ?? '—')}
                  </td>
                </tr>
              )
            })}
            {!results.length && (
              <tr><td colSpan={4} className="py-6 text-center text-faint">No scorecard yet — run the evaluation.</td></tr>
            )}
          </tbody>
        </table>
      </Panel>
    </div>
  )
}
