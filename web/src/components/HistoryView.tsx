import { useEffect, useRef, useState } from 'react'
import { fetchHistory, fetchJob, type HistoryRow, type JobState } from '../api'
import { taskLabel } from '../labels'
import { Panel } from './Console'
import Results from './Results'

export default function HistoryView() {
  const [rows, setRows] = useState<HistoryRow[]>([])
  const [sel, setSel] = useState<JobState | null>(null)
  const [filter, setFilter] = useState('')
  const resultsRef = useRef<HTMLDivElement>(null)

  useEffect(() => { refresh() }, [])

  function refresh() { fetchHistory(50).then(setRows).catch(() => {}) }

  async function open(jobId: string) {
    const st = await fetchJob(jobId)
    if (st.status === 'done') {
      setSel(st)
      // auto-scroll to the results after a short delay for render
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 100)
    }
  }

  const visible = rows.filter(r => !filter || r.selected_task.includes(filter))

  return (
    <div className="space-y-6">
      <Panel title="Run history">
        <div className="mb-3 flex flex-wrap gap-1.5">
          {['', 'single_vqa', 'captioning', 'grounding', 'change_vqa', 'change_analysis', 'optical_sar'].map(t => (
            <button key={t || 'all'} onClick={() => setFilter(t)}
              className={`rounded-full border px-2.5 py-0.5 text-[14px] transition-colors ${
                filter === t ? 'border-accent/60 bg-accent/10 text-accent'
                             : 'border-line text-muted hover:text-body'}`}>
              {t ? taskLabel(t) : 'all tasks'}
            </button>
          ))}
          <button onClick={refresh}
            className="ml-auto rounded border border-line px-2.5 py-0.5 text-[14px] text-muted hover:text-body">↻ refresh</button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-[15px]">
            <thead>
              <tr className="border-b border-line text-[13px] uppercase tracking-wider text-faint">
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
                  <td className="py-2 pr-3 font-mono text-[13px] text-muted">{r.created_at.slice(5, 16)}</td>
                  <td className="py-2 pr-3 text-[15px] text-accent">{taskLabel(r.selected_task) !== '—' ? taskLabel(r.selected_task) : r.status}</td>
                  <td className="max-w-[220px] truncate py-2 pr-3 text-muted">{r.query}</td>
                  <td className="max-w-[280px] truncate py-2 pr-3 text-body">{r.answer}</td>
                  <td className="py-2 pr-3 text-right font-mono text-[13px] text-muted">{r.confidence?.toFixed?.(2)}</td>
                  <td className="py-2 text-right">
                    <button onClick={() => open(r.job_id)} className="text-accent hover:underline">view</button>
                  </td>
                </tr>
              ))}
              {!visible.length && (
                <tr><td colSpan={6} className="py-6 text-center text-faint">No runs yet.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </Panel>

      {sel?.result && (
        <div ref={resultsRef}>
          <Results result={sel.result} />
        </div>
      )}
    </div>
  )
}
