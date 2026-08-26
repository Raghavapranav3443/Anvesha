const ROWS = [
  { task: 'Visual Q&A', approach: 'per-type specialist heads + CORAL counting',
    data: 'RSVQA-LR · 54k triplets', score: '0.70 exact-match' },
  { task: 'Scene description', approach: 'plan-conditioned transformer decoder',
    data: 'BigEarthNet.txt captions', score: '0.28 multi-ref BLEU' },
  { task: 'Region highlighting', approach: 'spectral-index reasoning — NDWI · ExG · double-bounce',
    data: 'no training — interpretable', score: 'boxes + masks' },
  { task: 'Change detection', approach: 'Siamese FPN-lite · tiled inference',
    data: 'LEVIR-CD', score: '0.60 IoU · 0.75 F1' },
  { task: 'Optical + SAR fusion', approach: 'dual-branch · dB-aware SAR path',
    data: 'BEN v2 S1+S2 pairs', score: '0.85 label recall' },
  { task: 'Investigation', approach: 'multi-step agent — change → water → impact',
    data: 'end-to-end', score: 'quantified impact' },
]

/** Specialist manifest styled as a payload spec sheet — no cards. */
export default function ManifestTable() {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] border-collapse text-left">
        <thead>
          <tr className="border-b border-line font-mono text-[12px] uppercase tracking-[.22em] text-faint">
            <th className="py-3 pr-4 font-medium">Task</th>
            <th className="py-3 pr-4 font-medium">Approach</th>
            <th className="py-3 pr-4 font-medium">Trained on</th>
            <th className="py-3 text-right font-medium">Measured</th>
          </tr>
        </thead>
        <tbody>
          {ROWS.map((r) => (
            <tr key={r.task} className="border-b border-line transition-colors hover:bg-elev">
              <td className="py-3.5 pr-4 text-[16px] font-medium text-body">{r.task}</td>
              <td className="py-3.5 pr-4 text-[14.5px] text-muted">{r.approach}</td>
              <td className="py-3.5 pr-4 font-mono text-[13.5px] text-faint">{r.data}</td>
              <td className="py-3.5 text-right font-mono text-[14px] text-accent">{r.score}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
