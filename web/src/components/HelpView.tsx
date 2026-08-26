import { useState } from 'react'
import { Panel } from './Console'

const GLOSSARY = [
  ['ANVESHA', 'From Sanskrit आन्वेषा — "search, investigation, discovery". Named for what it does: investigate Earth observation imagery.'],
  ['Agentic analysis', 'The system doesn\'t apply one model — it plans a sequence of specialist models, runs them, and fuses their outputs. Every step is visible in the execution trace.'],
  ['Investigation Mode', 'A full autonomous workflow: change detection → water extraction → impact quantification → zone ranking → analyst recommendations, in one click.'],
  ['Impact quantification', 'Change converted to decision-ready numbers: hectares affected, % within buffer distances of water, ranked priority zones.'],
  ['Calibrated confidence', 'Confidence scores fit against held-out data (temperature scaling) — 0.8 genuinely means ~80% historical reliability.'],
  ['GeoTIFF awareness', 'Geographic reference, resolution and CRS are preserved and used — areas are computed in real-world units, overlays export as GeoJSON.'],
  ['Change mask', 'Pixel-level map of where the surface changed between two dates. Exportable as a GeoTIFF for GIS comparison against reference annotations.'],
  ['Execution trace', 'The observable audit trail: validation, query interpretation, tool selection, execution with parameters and timings.'],
  ['SAR', 'Synthetic Aperture Radar — works through clouds and at night. Calm water appears dark; urban structures appear bright.'],
  ['NDVI / NDWI / ExG', 'Spectral indices that highlight vegetation, water, and greenness respectively — the interpretable evidence behind grounding.'],
  ['GSD', 'Ground Sample Distance — the real-world size of one pixel in metres.'],
  ['Bi-temporal pair', 'Two images of the same place at different dates — the basis of change detection.'],
]

const CAPABILITIES = [
  ['🛰️ Console', 'Upload imagery and ask questions. Query mode for direct answers, Investigation Mode for a full multi-step analysis with quantified findings.'],
  ['🗂️ History', 'Every run is persisted in a local database. Reopen past analyses, filter by task, compare two runs side by side.'],
  ['📊 Evaluation', 'The reproducible benchmark scorecard — the same numbers reported in the README, re-runnable live from the browser.'],
  ['🧠 Provenance', 'Which models are loaded, what public data they were fine-tuned on, and their measured validation metrics.'],
  ['🗺️ Map & exports', 'Georeferenced runs render on a map as GeoJSON overlays; change masks export as GeoTIFF; reports as PDF/Markdown/JSON.'],
  ['❓ Click-to-query', 'Drag a box on any single image and ask what\'s inside it — the region is cropped and analysed on the spot.'],
]

const PIPELINE = [
  ['Validate inputs', 'Formats, band counts, sensor type, CRS and co-registration geometry are checked before any model runs.'],
  ['Interpret the query', 'Your question is classified into a task (VQA, captioning, grounding, change analysis, fusion or investigation).'],
  ['Select specialists', 'A tool registry matches the task to fine-tuned remote-sensing models.'],
  ['Execute & fuse', 'Specialists run in sequence; outputs, timings and parameters are recorded.'],
  ['Report with evidence', 'Answer + calibrated confidence + visual overlays + downloadable audit trail.'],
]

export default function HelpView({ onStart }: { onStart?: () => void }) {
  const [open, setOpen] = useState<string | null>(null)
  return (
    <div className="space-y-6">
      <Panel title="How Anvesha works">
        <p className="max-w-4xl text-[19px] leading-relaxed text-muted">
          You ask questions in plain language. An agent validates your imagery,
          interprets the question, selects fine-tuned remote-sensing specialist
          models, runs them, and fuses their outputs into an evidence-grounded
          answer. You never need to know the models, thresholds or GIS steps.
        </p>
        <div className="mt-4 grid gap-2.5 md:grid-cols-5">
          {PIPELINE.map(([t, d], i) => (
            <div key={t} className="rounded-lg border border-line bg-elev p-3">
              <div className="mb-1 font-mono text-[14px] text-accent">STEP {i + 1}</div>
              <div className="text-[18px] font-semibold text-body">{t}</div>
              <div className="mt-1 text-[16.5px] leading-relaxed text-muted">{d}</div>
            </div>
          ))}
        </div>
        {onStart && (
          <button onClick={onStart}
            className="mt-4 rounded-lg bg-accent px-5 py-2 text-base font-semibold text-white hover:bg-accent-dim">
            Open the console
          </button>
        )}
      </Panel>

      <Panel title="What each part does">
        <div className="grid gap-2.5 md:grid-cols-2">
          {CAPABILITIES.map(([t, d]) => (
            <div key={t} className="rounded-lg border border-line bg-elev p-3.5">
              <div className="text-[17.5px] font-semibold text-body">{t}</div>
              <div className="mt-1 text-[17px] leading-relaxed text-muted">{d}</div>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Glossary — tap a term to expand">
        <div className="grid gap-2 md:grid-cols-2">
          {GLOSSARY.map(([t, d]) => (
            <button key={t} onClick={() => setOpen(open === t ? null : t)}
              className={`rounded-lg border px-3.5 py-2.5 text-left transition-colors ${
                open === t ? 'border-accent/60 bg-accent-soft' : 'border-line bg-elev hover:border-muted/50'}`}>
              <div className="text-[17.5px] font-semibold text-body">{t}</div>
              {open === t && <div className="mt-1 text-[17px] leading-relaxed text-muted">{d}</div>}
            </button>
          ))}
        </div>
      </Panel>
    </div>
  )
}
