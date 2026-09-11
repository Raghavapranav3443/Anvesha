// C8 — judge-run checklist: 5 greens, each bound to a real History run_id.
import { useEffect, useState } from 'react'
import { fetchHistory, type HistoryRow } from '../api'

interface ChecklistItem {
  id: string
  label: string
  hint: string
  match: (r: HistoryRow) => boolean
}

const ITEMS: ChecklistItem[] = [
  { id: 'vqa', label: 'Single VQA', hint: 'Ask one question of one image',
    match: r => r.selected_task === 'single_vqa' },
  { id: 'caption', label: 'Caption / Ground', hint: 'Describe or highlight a region',
    match: r => r.selected_task === 'captioning' || r.selected_task === 'grounding' },
  { id: 'change', label: 'Change', hint: 'Compare or describe two dates (change_analysis / change_description)',
    match: r => r.selected_task === 'change_vqa' || r.selected_task === 'change_analysis' || r.selected_task === 'change_description' },
  { id: 'fusion', label: 'Optical+SAR fusion', hint: 'Multi-modal agreement map',
    match: r => r.selected_task === 'optical_sar' },
  { id: 'investigation', label: 'Orchestration', hint: 'Investigation chain',
    match: r => r.selected_task === 'investigation' },
]

export default function JudgeRun() {
  const [history, setHistory] = useState<HistoryRow[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true
    fetchHistory(50).then(h => { if (alive) { setHistory(h); setLoading(false) } })
    return () => { alive = false }
  }, [])

  const greens = ITEMS.filter(it => history.some(it.match)).length
  const total = ITEMS.length

  return (
    <div className="rounded-xl border border-line bg-panel p-5">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <div className="text-[15px] font-semibold text-body">Judge-run checklist</div>
          <div className="text-[12px] text-muted">5 mandatory workflows — each tied to a real run_id</div>
        </div>
        <div className={`rounded-full px-3 py-1 text-[12px] font-semibold ${
          greens === total ? 'bg-good/20 text-good' : 'bg-warn/20 text-warn'
        }`}>
          {greens}/{total} green{greens === total ? ' — all workflows live' : ' — keep going'}
        </div>
      </div>
      <ul className="space-y-2">
        {ITEMS.map(it => {
          const hit = history.find(it.match)
          return (
            <li key={it.id} className={`flex items-center gap-3 rounded-lg border px-3 py-2 ${
              hit ? 'border-good/30 bg-good/5' : 'border-line bg-elev'
            }`}>
              <span className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-bold ${
                hit ? 'bg-good text-white' : 'bg-line text-muted'
              }`}>{hit ? '✓' : '○'}</span>
              <div className="min-w-0 flex-1">
                <div className="text-[13.5px] font-medium text-body">{it.label}</div>
                <div className="truncate text-[11.5px] text-muted">{it.hint}</div>
              </div>
              {hit && (
                <div className="shrink-0 text-right">
                  <div className="font-mono text-[12px] text-good">run {hit.run_id?.slice(-8) || hit.job_id?.slice(-8)}</div>
                  <div className="text-[11px] text-muted">conf {((hit.confidence ?? 0) * 100).toFixed(0)}%</div>
                </div>
              )}
            </li>
          )
        })}
      </ul>
      {loading && <div className="mt-2 text-[11px] text-muted">Loading history…</div>}
    </div>
  )
}

