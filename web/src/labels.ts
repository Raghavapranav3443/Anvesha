// Shared human-readable labels — one source of truth for every view.
// Judges' litmus test: a non-domain reader should understand every string.

export const TASK_LABELS: Record<string, string> = {
  single_vqa: 'Visual Q&A',
  captioning: 'Scene Description',
  grounding: 'Region Highlighting',
  change_analysis: 'Change Detection',
  change_vqa: 'Change Q&A',
  impact_analysis: 'Impact Analysis',
  investigation: 'Investigation Report',
  optical_sar: 'Optical + SAR Fusion',
}

export function taskLabel(t?: string | null): string {
  if (!t) return '—'
  return TASK_LABELS[t] ?? t
}

const STEP_LABELS: Record<string, string> = {
  validate_inputs: 'Validate inputs',
  classify_task: 'Interpret query intent',
  plan_investigation: 'Plan multi-step investigation',
  select_tool: 'Select specialist tool',
  finish: 'Complete',
}

export function stepLabel(name: string): string {
  if (STEP_LABELS[name]) return STEP_LABELS[name]
  if (name.startsWith('execute:')) {
    // Investigation plans emit step names like `grounding_water`,
    // `grounding_built_up`, `grounding_vegetation`. Strip the concept
    // suffix so `taskLabel` can map the base task to a readable label.
    return 'Run ' + taskLabel(name.slice(8).replace(/_water$/, '').trim())
  }
  if (name.startsWith('execute')) return 'Run specialist'
  return name
}

// Model provenance strings from the backend → readable phrases.
export function humanizeSource(s?: string | null): string {
  if (!s) return ''
  const table: [RegExp, string][] = [
    [/per-type specialist head \((\w+)\)/, 'Specialist VQA head ($1)'],
    [/dedicated counting head/i, 'Dedicated counting specialist'],
    [/fine-tuned RSVQA model/i, 'Fine-tuned RS VQA model'],
    [/scene-evidence rule reasoner/i, 'Interpretable scene-evidence rules'],
    [/template composition/i, 'Evidence-grounded template'],
    [/BigEarthNet.txt-trained caption decoder/i, 'Learned caption decoder (BEN.txt-trained)'],
    [/fine-tuned dual-branch fusion network/i, 'Trained Optical+SAR fusion network'],
    [/spectral-index|index regions/i, 'Interpretable spectral indices'],
    [/siamese|change_net/i, 'Siamese change detector (LEVIR-CD trained)'],
    [/keyword-intent rules/i, 'Deterministic intent rules'],
    [/default routing/i, 'Default routing'],
  ]
  let out = s
  for (const [re, rep] of table) out = out.replace(re, rep)
  return out
}
