import { useEffect, useState } from 'react'
import { fetchHistory, fetchJob, type HistoryRow, type JobState } from '../api'
import { Panel } from './Console'
import Results from './Results'

export default function HistoryView() {
  const [rows, setRows] = useState<HistoryRow[]>([])
  const [sel, setSel] = useState<JobState | null>(null)
  const [compare, setCompare] = useState<string[]>([])
  const [cmpStates, setCmpStates] = useState<JobState[] | null>(null)
  const [filter, setFilter] = useState('')

  useEffect(() => { refresh() }, [])

  function refresh() { fetchHistory(50).then(setRows).catch(() => {}) }

  async function open(jobId: string) {
    const st = await fetchJob(jobId)
    if (st.status === 'done') { setSel(st); window.scrollTo({ top: 0, behavior: 'smooth' }) }
  }

  async function compareRuns() {
    const sts = await Promise.all(compare.map(fetchJob))
    setCmpStates(sts.filter(s => s.status === 'done'))
  }

  const visible = rows.filter(r => !filter || r.selected_task.includes(filter))

  return (
    <div className="space-y-6">
      {cmpStates && cmpStates.length === 2 && (
        <Panel title="Run comparison">
          <div className="grid grid-cols-2 gap-4">
            {cmpStates.map(s => (
              <div key={s.job_id} className="rounded-lg border border-line p-4">
                <div className="font-mono text-[11px] text-muted">{s.job_id}</div>
                <div className="mt-1 text-[13px] text-accent">{s.result?.selected_task}</div>
                <div className="mt-2 text-sm text-body">{s.result?.answer?.slice(0, 220)}</div>
                <div className="mt-2 font-mono text-xs text-muted">confidence {s.result?.confidence}</div>
              </div>
            ))}
          </div>
        </Panel>
      )}

      <Panel title="Run history (SQLite-persisted)">
        <div className="mb-3 flex flex-wrap gap-1.5">
          {['', 'single_vqa', 'captioning', 'grounding', 'change_vqa', 'change_analysis', 'optical_sar'].map(t => (
            <button key={t || 'all'} onClick={() => setFilter(t)}
              className={`rounded-full border px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-wide transition-colors ${
                filter === t ? 'border-accent/60 bg-accent/10 text-accent'
                             : 'border-line text-muted hover:text-body'}`}>
              {t || 'all'}
            </button>
          ))}
          <button onClick={refresh}
            className="ml-auto rounded border border-line px-2.5 py-0.5 text-[11px] text-muted hover:text-body">↻ refresh</button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-[12.5px]">
            <thead>
              <tr className="border-b border-line text-[10.5px] uppercase tracking-wider text-faint">
                <th className="py-2 pr-3 font-medium">When</th>
                <th className="py-2 pr-3 font-medium">Task</th>
                <th className="py-2 pr-3 font-medium">Query</th>
                <th className="py-2 pr-3 font-medium">Answer</th>
                <th className="py-2 pr-3 font-medium text-right">Conf</th>
                <th className="py-2 text-right font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {visible.map(r => (
                <tr key={r.job_id} className="border-b border-line/60 hover:bg-elev">
                  <td className="py-2 pr-3 font-mono text-[11px] text-muted">{r.created_at.slice(5, 16)}</td>
                  <td className="py-2 pr-3 font-mono text-[11px] text-accent">{r.selected_task || r.status}</td>
                  <td className="max-w-[220px] truncate py-2 pr-3 text-muted">{r.query}</td>
                  <td className="max-w-[280px] truncate py-2 pr-3 text-body">{r.answer}</td>
                  <td className="py-2 pr-3 text-right font-mono text-[11px] text-muted">{r.confidence?.toFixed?.(2)}</td>
                  <td className="py-2 text-right">
                    <button onClick={() => open(r.job_id)} className="mr-2 text-accent hover:underline">view</button>
                    <input type="checkbox" className="accent-[#4C8DF6]"
                      checked={compare.includes(r.job_id)}
                      onChange={() => setCompare(c =>
                        c.includes(r.job_id) ? c.filter(x => x !== r.job_id)
                                             : [...c, r.job_id].slice(-2))} />
                  </td>
                </tr>
              ))}
              {!visible.length && (
                <tr><td colSpan={6} className="py-6 text-center text-faint">No runs yet.</td></tr>
              )}
            </tbody>
          </table>
        </div>
        {compare.length === 2 && (
          <button onClick={compareRuns}
            className="mt-3 rounded-lg bg-accent px-4 py-1.5 text-xs font-semibold text-white hover:bg-accent-dim">
            Compare selected runs
          </button>
        )}
      </Panel>

      {sel?.result && <Results result={sel.result} />}
    </div>
  )
}
