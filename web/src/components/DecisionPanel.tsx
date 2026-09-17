// Decision layer — the final stage: "what do I do with this analysis?"
//
// Renders the frozen `decision` dict emitted by satquery/decision. Written for a
// non-expert: the headline says what to do, not what was computed. The
// "what this cannot tell you" section is shown as prominently as the action,
// because a recommendation without its limits is how a tool misleads someone.
//
// Every field is optional-safe: an older run without a decision record simply
// renders nothing.

type Decision = {
  schema?: string
  outcome?: string
  domain?: string
  trust?: {
    value?: number | null
    band?: string
    source?: string
    calibrated?: boolean
    limiting_factor?: string
    band_evidence?: {
      claimed?: number | null
      observed_accuracy?: number | null
      n?: number | null
    } | null
  }
  sensitivity?: { available?: boolean; plain?: string }
  detection_floor?: { gsd_m?: number | null; hectares?: number | null; plain?: string }
  // The authority lane: thematic layers ISRO's own map service shows for this
  // same area, verified present against a control render. Kept separate from our
  // own measurement so "what we found" and "what the official record says" are
  // never read as one claim.
  authority?: {
    available?: boolean
    themes?: string[]
    sentences?: string[]
    note?: string
  }
  advice?: {
    headline?: string
    outcome_label?: string
    what_we_found?: string
    what_it_means?: string
    what_to_do?: string
    who_to_tell?: string
    how_far_to_trust?: string
    what_we_cannot_tell?: string[]
    also_true?: string[]
    result_note?: string
    what_the_authority_says?: string
    authority_available?: boolean
  }
}

const OUTCOME_STYLE: Record<string, { ring: string; text: string; label: string }> = {
  act: { ring: 'border-emerald-500/50 bg-emerald-500/10', text: 'text-emerald-300', label: 'Act on this' },
  verify_first: { ring: 'border-amber-500/50 bg-amber-500/10', text: 'text-amber-300', label: 'Check before acting' },
  monitor: { ring: 'border-sky-500/50 bg-sky-500/10', text: 'text-sky-300', label: 'Worth watching' },
  no_action: { ring: 'border-line bg-elev', text: 'text-muted', label: 'No action needed' },
  insufficient_evidence: { ring: 'border-rose-500/40 bg-rose-500/5', text: 'text-rose-300', label: 'Cannot conclude' },
}

function Section({ title, body }: { title: string; body?: string }) {
  if (!body) return null
  return (
    <div>
      <div className="text-sm font-semibold uppercase tracking-wide text-faint">{title}</div>
      <p className="mt-1 leading-relaxed text-body">{body}</p>
    </div>
  )
}

export default function DecisionPanel({ decision }: { decision?: Decision }) {
  if (!decision?.advice) return null
  const a = decision.advice
  const style = OUTCOME_STYLE[decision.outcome ?? ''] ?? OUTCOME_STYLE.no_action
  const trust = decision.trust ?? {}
  const ev = trust.band_evidence
  const cannot = (a.what_we_cannot_tell ?? []).filter(Boolean)

  return (
    <div className={`rounded-xl border ${style.ring} p-6`}>
      <div className="flex flex-wrap items-center gap-3">
        <span className={`rounded border border-current/30 px-2 py-0.5 text-sm font-semibold uppercase tracking-wide ${style.text}`}>
          {a.outcome_label ?? style.label}
        </span>
        {decision.domain && decision.domain !== 'general' && (
          <span className="rounded border border-line bg-panel px-2 py-0.5 font-mono text-sm text-muted">
            {decision.domain}
          </span>
        )}
        {trust.band && (
          <span className="rounded border border-line bg-panel px-2 py-0.5 text-sm text-muted">
            reliability: <span className="font-semibold text-body">{trust.band.replace(/_/g, ' ')}</span>
            {typeof trust.value === 'number' && ` · ${(trust.value * 100).toFixed(0)}%`}
          </span>
        )}
        {trust.calibrated === false && (
          <span className="rounded border border-amber-500/40 bg-amber-500/5 px-2 py-0.5 text-sm text-amber-300"
                title="This number has not been checked against known outcomes for this particular tool.">
            not validated
          </span>
        )}
      </div>

      <h3 className="mt-4 text-xl font-semibold leading-snug text-body">{a.headline}</h3>

      <div className="mt-4 space-y-4">
        <Section title="What we found" body={a.what_we_found} />
        <Section title="What it means" body={a.what_it_means} />
        <Section title="What to do" body={a.what_to_do} />
        <Section title="Who to tell" body={a.who_to_tell} />
        <Section title="How far to trust this" body={a.how_far_to_trust} />
      </div>

      {ev && typeof ev.observed_accuracy === 'number' && (
        <div className="mt-4 rounded-lg border border-line bg-panel/60 p-3">
          <div className="text-sm font-semibold uppercase tracking-wide text-faint">
            Measured on held-out data
          </div>
          <div className="mt-1 font-mono text-sm text-muted">
            we said {typeof ev.claimed === 'number' ? `${(ev.claimed * 100).toFixed(0)}%` : '—'}
            {' · '}we were right {(ev.observed_accuracy * 100).toFixed(0)}%
            {typeof ev.n === 'number' ? ` · ${ev.n} cases` : ''}
          </div>
        </div>
      )}

      {!!decision.authority?.available && (
        <div className="mt-3 rounded-lg border border-line bg-panel/60 p-3">
          <div className="text-sm font-semibold uppercase tracking-wide text-faint">
            What the official record says
          </div>
          <ul className="mt-1 list-disc space-y-1 pl-5 text-sm leading-relaxed text-muted">
            {(decision.authority.sentences ?? []).map((t, i) => <li key={i}>{t}</li>)}
          </ul>
          {decision.authority.note && (
            <p className="mt-2 text-xs leading-relaxed text-faint">{decision.authority.note}</p>
          )}
        </div>
      )}

      {decision.sensitivity?.plain && (
        <div className="mt-3 rounded-lg border border-line bg-panel/60 p-3">
          <div className="text-sm font-semibold uppercase tracking-wide text-faint">
            Margin on this conclusion
          </div>
          <p className="mt-1 text-sm leading-relaxed text-muted">{decision.sensitivity.plain}</p>
        </div>
      )}

      {!!a.also_true?.length && (
        <div className="mt-3">
          <div className="text-sm font-semibold uppercase tracking-wide text-faint">
            Also true for this area
          </div>
          <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-muted">
            {a.also_true.map((t, i) => <li key={i}>{t}</li>)}
          </ul>
        </div>
      )}

      {!!cannot.length && (
        <div className="mt-4 rounded-lg border border-line bg-panel/60 p-3">
          <div className="text-sm font-semibold uppercase tracking-wide text-faint">
            What this analysis cannot tell you
          </div>
          <ul className="mt-1 list-disc space-y-1 pl-5 text-sm leading-relaxed text-muted">
            {cannot.map((t, i) => <li key={i}>{t}</li>)}
          </ul>
        </div>
      )}

      {a.result_note && (
        <p className="mt-3 text-sm italic leading-relaxed text-faint">{a.result_note}</p>
      )}
    </div>
  )
}
