// C1 — boards view: pins over real run_ids (fail-closed, never invents evidence).
import { useEffect, useState } from 'react'
import { fetchBoards } from '../api'

interface Pin {
  run_id: string; kind: string; title: string; artifact: string
  why: string; task: string; confidence: number
}

const KIND_GLYPH: Record<string, string> = {
  overlay: '▦', table: '☷', geo: '◉', answer: '✎', '': '·'
}

export default function BoardsView() {
  const [doc, setDoc] = useState<{ boards_v: number; generated_at: string; pins: Pin[] } | null>(null)
  const [err, setErr] = useState('')

  useEffect(() => {
    fetchBoards().then(setDoc).catch(e => setErr(e.message))
  }, [])

  if (err) return <div className="text-[13px] text-warn">Boards unavailable: {err}</div>
  if (!doc) return <div className="text-[12px] text-muted">Loading boards…</div>
  return (
    <div>
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-[16px] font-semibold text-body">Mission boards</h2>
        <span className="font-mono text-[11px] text-muted">
          generated {doc.generated_at.slice(0, 19)} · {doc.pins.length} pins
        </span>
      </div>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {doc.pins.map(p => (
          <a key={p.run_id + p.kind} href={`#run:${p.run_id}`}
            className="group rounded-lg border border-line bg-elev p-3 transition hover:border-accent hover:bg-panel">
            <div className="mb-1 flex items-center gap-2">
              <span className="text-accent">{KIND_GLYPH[p.kind] || KIND_GLYPH['']}</span>
              <span className="truncate text-[13.5px] font-medium text-body">{p.title}</span>
            </div>
            <div className="truncate text-[11.5px] text-muted">{p.why}</div>
            <div className="mt-2 flex items-center justify-between">
              <span className="rounded bg-accent/10 px-1.5 py-0.5 font-mono text-[10px] uppercase text-accent">
                {p.kind}
              </span>
              <span className="font-mono text-[11px] text-muted">
                {(p.confidence * 100).toFixed(0)}%
              </span>
            </div>
          </a>
        ))}
      </div>
      {doc.pins.length === 0 && (
        <div className="text-[13px] text-muted">No pins yet — run the board builder.</div>
      )}
    </div>
  )
}

