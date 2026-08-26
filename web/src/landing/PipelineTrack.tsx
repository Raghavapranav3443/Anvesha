import { useEffect, useRef, useState } from 'react'

const STEPS = [
  { name: 'validate', desc: 'format · CRS · co-registration' },
  { name: 'interpret', desc: 'intent from plain language' },
  { name: 'route', desc: 'feasibility-checked specialist' },
  { name: 'execute', desc: 'fine-tuned models, timed' },
  { name: 'integrate', desc: 'answer + confidence + overlays' },
  { name: 'report', desc: 'auditable trace + GeoTIFF' },
]

/** The agentic pipeline as an orbital ground track: the path draws in and
 *  each node ignites (staggered) when the section scrolls into view. */
export default function PipelineTrack() {
  const ref = useRef<HTMLDivElement>(null)
  const [on, setOn] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const io = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) { setOn(true); io.disconnect() } },
      { threshold: 0.3 },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])

  const W = 1080
  const pts = STEPS.map((_, i) => ({
    x: 90 + (i * (W - 180)) / (STEPS.length - 1),
    y: i % 2 === 0 ? 128 : 92,
  }))
  const d = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ')

  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} 240`} className="w-full" role="img"
        aria-label="Agent pipeline: validate, interpret, route, execute, integrate, report">
        <path d={d} fill="none" stroke="#35c5f2" strokeOpacity="0.5" strokeWidth="1.5"
          strokeLinejoin="round" strokeDasharray="1600"
          strokeDashoffset={on ? 0 : 1600}
          style={{ transition: 'stroke-dashoffset 1.8s ease' }} />
        {pts.map((p, i) => (
          <g key={i}
            style={{ opacity: on ? 1 : 0.12, transition: `opacity .5s ease ${0.25 + i * 0.22}s` }}>
            <circle cx={p.x} cy={p.y} r="10" fill="none" stroke="#35c5f2" strokeOpacity="0.5" />
            <circle cx={p.x} cy={p.y} r="3.5" fill={i === STEPS.length - 1 ? '#7dd3fc' : '#35c5f2'} />
            <text x={p.x} y={p.y + (i % 2 === 0 ? -34 : 40)} textAnchor="middle"
              fontSize="16" letterSpacing="2"
              className="font-mono" fill="var(--c-text)">
              {STEPS[i].name.toUpperCase()}
            </text>
            <text x={p.x} y={p.y + (i % 2 === 0 ? -54 : 60)} textAnchor="middle"
              fontSize="13"
              className="hidden font-mono md:inline" fill="var(--c-muted)">
              {STEPS[i].desc}
            </text>
          </g>
        ))}
      </svg>
    </div>
  )
}
