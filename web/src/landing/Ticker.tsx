const ITEMS = [
  'confidence calibrated — T fit on held-out',
  'int8 torchscript −0.16% accuracy',
  '100/100 concurrent · p95 ≈ 7 s',
  'sqlite cache — identical query → instant',
  'offline-deployable · docker · no cloud',
  '63 tests green',
]

/** Mission-telemetry marquee. Pure CSS loop; static under reduced motion. */
export default function Ticker() {
  return (
    <div className="overflow-hidden border-b border-line py-3.5" aria-hidden>
      <div className="ticker-track flex w-max">
        {[...ITEMS, ...ITEMS].map((t, i) => (
          <span key={i} className="flex items-center whitespace-nowrap font-mono text-[13px] uppercase tracking-[.22em] text-muted">
            <span className="px-6">{t}</span>
            <span className="text-accent">▸</span>
          </span>
        ))}
      </div>
    </div>
  )
}
