// C4 — HonestyBanner: renders the frozen `honesty` dict emitted by the patch
// layer. Data-driven (honesty_copy.ts); never blocks the answer; amber theme.

import { rulesFor } from '../honesty_copy'

export default function HonestyBanner({ honesty }: {
  honesty?: {
    fallback_active?: boolean
    below_gate?: unknown[]
    pixel_space?: boolean
    limitation_refs?: string[]
  } | null
}) {
  const rules = rulesFor(honesty)
  if (rules.length === 0) return null
  return (
    <div className="space-y-2">
      {rules.map(({ rule, detail }, i) => (
        <div key={rule.when + i}
          className="rounded-lg border border-warn/40 bg-warn-soft px-3 py-2">
          <div className="text-[13.5px] font-medium text-warn">{rule.title}</div>
          <div className="mt-0.5 text-[13.5px] leading-snug text-body">
            {rule.text}{detail && (
              <span className="mt-1 block font-mono text-[12.5px] text-muted">{detail}</span>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}