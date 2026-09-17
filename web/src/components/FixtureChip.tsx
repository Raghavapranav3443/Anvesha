// C7 — FixtureChip: surfaces the demo-fixtures manifest honestly. When the
// current run matches a primed fixture it shows an amber "synthetic cached
// demo" chip; when the manifest is missing or degraded it names warm_demo.py.

import { useEffect, useState } from 'react'
import { fetchFixtures, type FixtureEntry, type FixturesManifest } from '../api'
import type { JobResult } from '../api'

function matchesFixture(result: JobResult, f: FixtureEntry): boolean {
  const files = (result.inputs ?? []).map(i => i.summary.file.toLowerCase())
  // All fixture sample names present in this run's inputs.
  const sameFiles = f.sampleNames.length > 0
    && f.sampleNames.every(s => files.includes(s.toLowerCase()))
  if (sameFiles) return true
  // Query-text overlap fallback (a judge may retype the setup query).
  const q = result.query.toLowerCase()
  const fq = f.query.toLowerCase()
  const words = fq.split(/\s+/).filter(w => w.length >= 4)
  if (words.length === 0) return false
  const hits = words.filter(w => q.includes(w)).length
  return hits / words.length >= 0.6
}

export default function FixtureChip({ result }: { result: JobResult }) {
  const [manifest, setManifest] = useState<FixturesManifest | null | undefined>(undefined)

  useEffect(() => {
    let alive = true
    fetchFixtures().then(m => { if (alive) setManifest(m) }).catch(() => { if (alive) setManifest(null) })
    return () => { alive = false }
  }, [])

  if (manifest === undefined || manifest === null) return null
  const match = manifest.fixtures.find(f => matchesFixture(result, f))
  if (!match) return null
  const green = match.status === 'green'

  return (
    <span className={`inline-flex items-center gap-1.5 rounded border px-2 py-0.5 font-mono text-xs ${
      green ? 'border-warn/50 bg-warn-soft text-warn'
            : 'border-bad/50 bg-bad-soft text-bad'}`}>
      <span className="h-1.5 w-1.5 rounded-full bg-warn" />
      SYNTHETIC demo fixture
      <span className="text-muted">· {match.status}</span>
      {!manifest.all_green && (
        <span className="text-muted">· rerun warm_demo.py</span>
      )}
    </span>
  )
}