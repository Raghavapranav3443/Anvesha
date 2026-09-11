// C4 — single source of honesty copy. Every rule maps to how it renders.
// All text is factual, non-blocking, and names the actual values from the
// run (never invented). Add a rule here and the banner picks it up; never
// embed copy in components.

export interface HonestyRule {
  when: 'fallback' | 'below_gate' | 'pixel_space' | 'limitation'
  title: string
  text: string
}

export const HONESTY_RULES: HonestyRule[] = [
  {
    when: 'fallback',
    title: 'Heuristic fallback active',
    text: 'One or more specialist components could not load trained weights on this '
      + 'machine and ran on an interpretable heuristic instead. Results remain '
      + 'auditable, but benchmark-precision claims do not apply to this run.',
  },
  {
    when: 'below_gate',
    title: 'Why no confident number',
    text: 'This answer\u2019s confidence falls below the published reliability gate, '
      + 'so treat the numerical claim as a strong hint rather than a measured '
      + 'quantity. The method and calibration size are shown.',
  },
  {
    when: 'pixel_space',
    title: 'Coordinates shown in pixel space',
    text: 'The input is not georeferenced (no CRS), so locations are reported in '
      + 'image coordinates — not guessed as latitude/longitude.',
  },
  {
    when: 'limitation',
    title: 'Known limits',
    text: 'Where a benchmark exists, the number follows the benchmark protocol. '
      + 'Where it does not, the method states the estimate explicitly.',
  },
]

export function rulesFor(honesty: {
  fallback_active?: boolean
  below_gate?: unknown[]
  pixel_space?: boolean
  limitation_refs?: string[]
} | null | undefined): { rule: HonestyRule; detail?: string }[] {
  if (!honesty) return []
  const out: { rule: HonestyRule; detail?: string }[] = []
  if (honesty.fallback_active) out.push({ rule: HONESTY_RULES[0] })
  const below = honesty.below_gate ?? []
  if (Array.isArray(below) && below.length > 0) {
    const parts = below.map((b: any) =>
      `${b.component ?? 'component'} conf ${((b.confidence ?? 0) * 100).toFixed(0)}%`)
    out.push({ rule: HONESTY_RULES[1], detail: parts.join(' · ') })
  }
  if (honesty.pixel_space) out.push({ rule: HONESTY_RULES[2] })
  const refs = honesty.limitation_refs ?? []
  if (Array.isArray(refs) && refs.length > 0) {
    out.push({ rule: HONESTY_RULES[3], detail: refs.slice(0, 3).join('; ') })
  }
  return out
}