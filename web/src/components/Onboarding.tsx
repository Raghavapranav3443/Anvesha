import { useState } from 'react'

const STEPS = [
  {
    icon: '🛰️',
    title: '1 · Bring your imagery',
    body: 'Upload one satellite image, a bi-temporal pair (two dates), or a co-registered optical + SAR pair. GeoTIFF, TIFF, PNG and JPEG are understood — geographic reference, band counts and sensor type are detected automatically.',
    demo: 'No imagery handy? Load a bundled demo sample from the sidebar.',
  },
  {
    icon: '💬',
    title: '2 · Ask in plain language',
    body: '"What changed between these two dates?", "Highlight the water body", "Is there a road?" — no GIS or remote-sensing vocabulary required. The agent interprets your question, picks the right specialist models and runs them.',
    demo: 'Toggle 🛰 Investigation Mode for a full multi-step analysis with quantified findings.',
  },
  {
    icon: '🔍',
    title: '3 · Get evidence, not just answers',
    body: 'Every answer ships with visual overlays, a confidence score, the exact models used and a step-by-step execution trace you can audit. Reports export as PDF, Markdown, JSON and GeoTIFF.',
    demo: 'Open the Evaluation tab any time to see measured benchmark scores.',
  },
]

export default function Onboarding({ onDone }: { onDone: () => void }) {
  const [step, setStep] = useState(0)
  const s = STEPS[step]

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/45 p-6 backdrop-blur-sm">
      <div className="fade-up w-full max-w-xl rounded-2xl border border-line bg-panel p-8 shadow-2xl">
        <div className="mb-1 flex items-center justify-between">
          <span className="font-mono text-[11px] uppercase tracking-[.18em] text-accent">
            Welcome to Anvesha
          </span>
          <span className="font-mono text-[11px] text-faint">{step + 1} / {STEPS.length}</span>
        </div>
        <h2 className="mb-4 text-2xl font-bold text-body">{s.icon} {s.title}</h2>
        <p className="text-[15.5px] leading-relaxed text-muted">{s.body}</p>
        <p className="mt-3 rounded-lg bg-elev px-3 py-2 text-[13.5px] italic text-body">{s.demo}</p>

        <div className="mt-6 flex items-center justify-between">
          <div className="flex gap-1.5">
            {STEPS.map((_, i) => (
              <span key={i} className={`h-1.5 w-6 rounded-full ${i === step ? 'bg-accent' : 'bg-line'}`} />
            ))}
          </div>
          <div className="flex gap-2">
            <button onClick={onDone} className="rounded-lg px-4 py-2 text-sm text-muted hover:text-body">Skip</button>
            <button
              onClick={() => (step < STEPS.length - 1 ? setStep(step + 1) : onDone())}
              className="rounded-lg bg-accent px-6 py-2 text-sm font-semibold text-white hover:bg-accent-dim">
              {step < STEPS.length - 1 ? 'Next' : 'Start exploring'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
