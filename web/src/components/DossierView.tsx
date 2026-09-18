// C2 — dossier view: head -> key_facts -> before/after -> timeline -> trace.
import { useEffect, useState } from 'react'
import { fetchDossier } from '../api'

export default function DossierView({ runId }: { runId: string }) {
  const [doc, setDoc] = useState<Record<string, unknown> | null>(null)
  const [err, setErr] = useState('')

  useEffect(() => {
    if (!runId) return
    fetchDossier(runId).then(setDoc).catch(e => setErr(e.message))
  }, [runId])

  if (!runId) return <div className="text-sm text-muted">No run selected.</div>
  if (err) return <div className="text-sm text-warn">Dossier unavailable: {err}</div>
  if (!doc) return <div className="text-xs text-muted">Loading dossier…</div>

  const head = (doc.head as Record<string, unknown>) || {}
  const facts = (doc.key_facts as string[]) || []
  const beforeAfter = (doc.before_after as Record<string, unknown>) || null
  const timeline = (doc.timeline as Record<string, unknown>[]) || []
  const reticles = (doc.reticles as Record<string, unknown>[]) || []

  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between">
        <h2 className="text-base font-semibold text-body">Dossier</h2>
        <span className="font-mono text-xs text-muted">run {(doc.run_id as string || '').slice(-8)}</span>
      </div>
      {typeof head.answer === 'string' && head.answer.length > 0 && (
        <p className="rounded-lg border border-accent/30 bg-accent/5 px-3 py-2 text-sm font-medium text-body">{head.answer}</p>
      )}
      {facts.length > 0 && (
        <div>
          <div className="mb-1 text-xs uppercase tracking-wider text-muted">Key facts</div>
          <ul className="list-inside list-disc space-y-0.5 text-sm text-body">
            {facts.map((f, i) => <li key={i}>{f}</li>)}
          </ul>
        </div>
      )}
      {beforeAfter && (
        <div className="rounded-lg border border-line p-3">
          <div className="text-xs uppercase tracking-wider text-muted">Change</div>
          <div className="mt-1 text-sm text-body">
            Changed area: <span className="font-mono">{String((beforeAfter.changed_area_ha as number || 0).toFixed(2))} ha</span>
          </div>
        </div>
      )}
      {timeline.length > 0 && (
        <div>
          <div className="mb-1 text-xs uppercase tracking-wider text-muted">Timeline</div>
          <div className="space-y-1">
            {timeline.map((e, i) => (
              <div key={i} className="flex items-center gap-2 text-sm">
                <span className="font-mono text-xs text-accent">
                  {(e.duration_ms as number || 0)}ms
                </span>
                <span className="text-body">{e.name as string}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {reticles.length > 0 && (
        <div>
          <div className="mb-1 text-xs uppercase tracking-wider text-muted">Localised regions</div>
          <div className="flex flex-wrap gap-2">
            {reticles.map((r, i) => (
              <span key={i} className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                r.verdict === 'confirmed' ? 'bg-good/20 text-good' :
                r.verdict === 'uncertain' ? 'bg-warn/20 text-warn' : 'bg-line text-muted'
              }`}>
                {r.verdict as string} ({r.x_pct as number}%, {r.y_pct as number}%)
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

