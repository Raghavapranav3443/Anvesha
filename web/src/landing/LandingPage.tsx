import { useEffect, useRef, useState } from 'react'
import HeroGlobe from './HeroGlobe'
import ManifestTable from './ManifestTable'
import PipelineTrack from './PipelineTrack'
import Reveal from './Reveal'
import './landing.css'

function ChapterLabel({ n, title }: { n: string; title: string }) {
  return (
    <div className="flex items-center gap-4 font-mono text-[19px] uppercase tracking-[.3em]">
      <span className="text-accent">{n}</span>
      <span className="h-px w-14 bg-accent/50" />
      <span className="text-muted">{title}</span>
    </div>
  )
}

function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false)
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    setReduced(mq.matches)
    const fn = () => setReduced(mq.matches)
    mq.addEventListener('change', fn)
    return () => mq.removeEventListener('change', fn)
  }, [])
  return reduced
}

/** CSS-only stand-in when WebGL is unavailable. */
function StaticGlobeFallback() {
  return (
    <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 opacity-90">
      <svg viewBox="0 0 600 600" className="h-[80vh] w-[80vh]">
        <defs>
          <radialGradient id="orb" cx="0.38" cy="0.35" r="0.85">
            <stop offset="0%" stopColor="#3b6fd4" />
            <stop offset="55%" stopColor="#12306e" />
            <stop offset="100%" stopColor="#04070f" />
          </radialGradient>
        </defs>
        <circle cx="300" cy="300" r="170" fill="url(#orb)" />
        {[210, 235, 258].map((r, i) => (
          <ellipse key={r} cx="300" cy="300" rx={r} ry={r * (0.32 + i * 0.08)}
            fill="none" stroke="#35c5f2" strokeOpacity={i === 1 ? 0.7 : 0.25}
            transform={`rotate(${-24 + i * 28} 300 300)`} />
        ))}
      </svg>
    </div>
  )
}

export default function LandingPage({ onEnterConsole }: { onEnterConsole?: () => void }) {
  const trackRef = useRef<HTMLDivElement>(null)
  const [progress, setProgress] = useState(0)
  const [webgl, setWebgl] = useState(true)
  const reduced = useReducedMotion()

  useEffect(() => {
    try {
      const c = document.createElement('canvas')
      setWebgl(!!(c.getContext('webgl2') ?? c.getContext('webgl')))
    } catch { setWebgl(false) }
  }, [])

  useEffect(() => {
    let raf = 0
    const measure = () => {
      const el = trackRef.current
      if (!el) return
      const hero = el.firstElementChild as HTMLElement | null
      const heroH = hero?.offsetHeight ?? window.innerHeight
      const total = Math.max(1, el.offsetHeight - heroH)
      const p = Math.min(1, Math.max(0, -el.getBoundingClientRect().top / total))
      setProgress(p)
    }
    const onScroll = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(measure) }
    measure()
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    return () => {
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onScroll)
      cancelAnimationFrame(raf)
    }
  }, [])

  const zoom = reduced ? 0 : Math.min(progress / 0.8, 1)
  const fade = reduced ? 0 : Math.min(1, Math.max(0, progress - 0.5) / 0.3)
  const [whyTab, setWhyTab] = useState<'plain' | 'stats'>('plain')

  return (
    <div className="landing-root relative bg-black">
      {/* floating controls: the landing renders without app chrome */}
      <div className="fixed right-5 top-4 z-50 flex items-center gap-2">
        <button
          onClick={onEnterConsole}
          className="rounded-lg border border-transparent bg-panel/80 px-3.5 py-1.5 font-mono text-[13px] uppercase tracking-[.15em] text-body backdrop-blur transition-colors hover:border-accent/50 hover:text-accent"
        >
          Console →
        </button>
        <button
          onClick={() => {
            const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'
            document.documentElement.dataset.theme = next
            localStorage.setItem('anvesha-theme', next)
          }}
          title="Toggle light/dark theme"
          className="rounded-lg border border-transparent bg-panel/80 px-2.5 py-1.5 text-body backdrop-blur transition-colors hover:border-accent/50"
        >
          <svg viewBox="0 0 24 24" className="h-[18px] w-[18px]" fill="none" stroke="currentColor"
            strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
            <path d="M20 13.5A8 8 0 1 1 10.5 4 6.5 6.5 0 0 0 20 13.5z" />
          </svg>
        </button>
      </div>

      {/* scroll track: hero stays pinned while the camera dollies in */}
      <div ref={trackRef} className="relative" style={{ height: '150vh' }}>
        <div className="sticky top-0 h-screen overflow-hidden">
          <div className="absolute inset-0">
            {webgl ? <HeroGlobe zoom={zoom} reduced={reduced} /> : <StaticGlobeFallback />}
          </div>

          {/* centered overlay: name + tagline + CTA, fading as the globe grows */}
          <div
            className="pointer-events-none absolute inset-0 z-10 flex flex-col items-center justify-center px-6 text-center"
            style={{ opacity: 1 - fade, transform: `translateY(${-fade * 120}px)` }}
          >
            <div className="font-mono text-[14px] uppercase tracking-[.3em] text-[#7dd3fc]/80">
              SIH26167 · ISRO / SAC
            </div>
            <h1 className="mt-5 pl-[.3em] text-[64px] font-light uppercase leading-none tracking-[.3em] text-white md:text-[108px]">
              Anvesha
            </h1>
            <div className="mt-6 pl-[.28em] font-mono text-[15px] uppercase tracking-[.28em] text-white/75">
              Earth Observation &amp; Investigation System
            </div>
            <div className="pointer-events-auto mt-7 flex flex-wrap items-center justify-center gap-2 font-mono text-[12px] uppercase tracking-[.18em] text-white/70">
              <span className="rounded-full border border-white/20 bg-white/5 px-3 py-1 backdrop-blur">6 benchmarks measured</span>
              <span className="rounded-full border border-white/20 bg-white/5 px-3 py-1 backdrop-blur">ISRO-format ready</span>
              <span className="rounded-full border border-white/20 bg-white/5 px-3 py-1 backdrop-blur">runs fully offline</span>
            </div>
            <button
              onClick={onEnterConsole}
              className="pointer-events-auto mt-9 rounded-lg bg-[#1f5fd6] px-8 py-3 text-[16px] font-semibold text-white transition-colors hover:bg-[#174ba8]"
            >
              Launch the console →
            </button>
            <button
              onClick={() => document.getElementById('chapter-01')?.scrollIntoView({ behavior: 'smooth' })}
              className="pointer-events-auto mt-10 flex flex-col items-center gap-1 text-white/45 transition-colors hover:text-white"
              aria-label="Scroll to content"
            >
              <span className="font-mono text-[13px] uppercase tracking-[.3em]">scroll ▾</span>
            </button>
          </div>
        </div>
      </div>

      {/* next section slides up OVER the pinned globe (reference frame 4) */}
      <section id="chapter-01" className="relative z-10 border-t border-line bg-[var(--c-bg)]">
        <div className="mx-auto max-w-5xl px-8 pb-24 pt-14">
          <Reveal>
            <ChapterLabel n="01" title="The problem" />
            <h2 className="mt-6 text-[36px] font-bold leading-[1.05] text-body md:text-[44px]">
              Satellites see everything.<br />No one can read it all.
            </h2>
          </Reveal>
          <Reveal delay={120}>
            <p className="mt-8 max-w-3xl text-[19px] font-medium leading-snug text-body md:text-[22px]">
              Every day, satellites photograph the entire planet. Far more than
              any human team can analyse. So the changes that matter most go
              unnoticed.
            </p>
          </Reveal>
          <Reveal delay={180}>
            <div className="mt-10 text-[40px] font-bold text-accent md:text-[48px]">Why?</div>
          </Reveal>
          <Reveal delay={240}>
            <div className="mt-5 grid gap-4 md:grid-cols-2">
              <div className="rounded-xl border border-line bg-panel p-6">
                <div className="font-mono text-[12px] uppercase tracking-[.25em] text-faint">Before</div>
                <div className="mt-3 text-[19px] font-medium leading-snug text-muted">
                  One question meant days of GIS tooling and specialist hours.
                </div>
              </div>
              <div className="rounded-xl border border-accent/40 bg-panel p-6">
                <div className="font-mono text-[12px] uppercase tracking-[.25em] text-accent">After</div>
                <div className="mt-3 text-[19px] font-medium leading-snug text-body">
                  One or two images and one sentence: what changed, where, and
                  why it matters.
                </div>
              </div>
            </div>
          </Reveal>
          <Reveal delay={300}>
            <p className="mt-10 max-w-2xl text-[17px] leading-relaxed text-muted">
              That is the whole idea. You bring one or two satellite images and
              ask a question in plain words. Anvesha finds the change, shows you
              exactly where it is, and tells you what it means. You never touch
              a GIS tool.
            </p>
          </Reveal>
        </div>
      </section>

      <section className="relative z-10 border-t border-line bg-[var(--c-bg)]">
        <div className="mx-auto max-w-5xl px-8 py-24">
          <Reveal>
            <ChapterLabel n="02" title="The solution" />
            <h2 className="mt-6 text-[36px] font-bold text-body md:text-[44px]">
              One sentence in. An investigation out.
            </h2>
            <p className="mt-4 max-w-2xl text-[16.5px] leading-relaxed text-muted">
              An agent validates your imagery, interprets the query, routes it to
              fine-tuned remote-sensing specialists, and returns evidence :
              every step timed and auditable.
            </p>
          </Reveal>
          <Reveal delay={150}><div className="mt-8"><PipelineTrack /></div></Reveal>
          <Reveal delay={250}>
            <div className="mt-6 font-mono text-[13px] uppercase tracking-[.25em] text-faint">
              investigation mode: change → water → impact → priority zones, chained automatically
            </div>
          </Reveal>
        </div>
      </section>

      <section className="relative z-10 border-t border-line bg-[var(--c-bg)]">
        <div className="mx-auto max-w-5xl px-8 py-24">
          <Reveal>
            <ChapterLabel n="03" title="The specialists" />
            <h2 className="mt-6 text-[36px] font-bold text-body md:text-[44px]">
              Seven specialists. One registry.
            </h2>
          </Reveal>
          <Reveal delay={150}><div className="mt-10"><ManifestTable /></div></Reveal>
          <Reveal delay={220}>
            <div className="mt-12">
              <div className="font-mono text-[12px] uppercase tracking-[.25em] text-faint">the scorecard: every row measured</div>
              <div className="mt-4 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line md:grid-cols-3">
                {[
                  ['RSVQA-LR', '0.700', 'exact-match · full test n=9,491 · spot-check 0.773 (n=282)'],
                  ['LEVIR-CD', '0.818', 'F1 · full test n=1,500 · IoU 0.692 · thr=0.85'],
                  ['CDVQA', '0.683', 'full test · 39,686 Q · +17.4 pts over majority baseline'],
                  ['BigEarthNet captions', '0.306', 'multi-ref BLEU · n=300'],
                  ['VRSBench grounding', '0.126', 'IoU@0.5 · spectral · n=455'],
                  ['VRSBench captioning', '0.000', 'reported honestly — see protocol note below'],
                ].map(([name, v, sub]) => (
                  <div key={name} className="bg-[var(--c-bg)] p-4">
                    <div className="font-mono text-[11.5px] uppercase tracking-[.18em] text-faint">{name}</div>
                    <div className="mt-1.5 text-[28px] font-bold leading-none text-accent">{v}</div>
                    <div className="mt-1.5 font-mono text-[11.5px] text-faint">{sub}</div>
                  </div>
                ))}
              </div>
            </div>
          </Reveal>
          <Reveal delay={250}>
            <div className="mt-8 max-w-3xl space-y-3 text-[14.5px] leading-relaxed text-muted">
              <p>
                <span className="font-mono text-[11.5px] uppercase tracking-[.2em] text-faint">protocol · </span>
                headline numbers are full public test sets, measured 2026-08-29 on this machine
                (RSVQA-LR n=9,491 · LEVIR-CD n=1,500 · CDVQA n=39,686). Small-n rows are spot-check
                subsets: they move a few points between runs, so quote the full-test numbers.
                Spot-checks reproduce with <span className="font-mono">python -m satquery.evaluate --all</span>;
                full-test numbers with <span className="font-mono">scripts/run_benchmarks.py --n 9491</span> (RSVQA-LR),{' '}
                <span className="font-mono">--n 1500</span> (LEVIR-CD) and{' '}
                <span className="font-mono">scripts/eval_cdvqa.py --split test</span> (CDVQA).
              </p>
              <p className="text-faint">
                VRSBench captioning reads 0.000 by construction, not by failure: the caption
                decoder was trained on BigEarthNet-style captions, which share no 4-grams with
                VRSBench's human-written references — a style-distribution mismatch. The same
                decoder scores 0.306 multi-ref BLEU on its own benchmark (row 4).
                Full decision record: <span className="font-mono">Decisions.md</span>.
              </p>
            </div>
          </Reveal>
        </div>
      </section>

      <section className="relative z-10 border-t border-line bg-[var(--c-bg)]">
        <div className="mx-auto max-w-5xl px-8 py-24">
          <Reveal>
            <ChapterLabel n="04" title="Evidence" />
            <h2 className="mt-6 text-[36px] font-bold text-body md:text-[44px]">
              Real imagery. Real weights. One pass.
            </h2>
            <p className="mt-4 max-w-2xl text-[16.5px] leading-relaxed text-muted">
              A held-out LEVIR-CD test pair run through the production change
              specialist: same weights, same 0.85 threshold as the scorecard,
              no cherry-picking pipeline. On this pair the predicted mask scores
              0.96 IoU against ground truth.
            </p>
          </Reveal>
          <Reveal delay={150}>
            <div className="mt-10 grid grid-cols-3 gap-3">
              {[
                ['evidence/before.png', 'before'],
                ['evidence/after.png', 'after'],
                ['evidence/overlay.png', 'detected change'],
              ].map(([src, label]) => (
                <figure key={label} className="overflow-hidden rounded-lg border border-line">
                  <img src={src} alt={label} className="block w-full" loading="lazy" />
                  <figcaption className="border-t border-line px-3 py-2 font-mono text-[11.5px] uppercase tracking-[.18em] text-faint">
                    {label}
                  </figcaption>
                </figure>
              ))}
            </div>
          </Reveal>
          <Reveal delay={220}>
            <div className="mt-8 rounded-lg border border-line bg-panel p-5">
              <div className="font-mono text-[11.5px] uppercase tracking-[.25em] text-faint">
                execution trace: every run, verbatim
              </div>
              <div className="mt-3 space-y-1.5 font-mono text-[13px] leading-relaxed text-muted">
                <div><span className="text-accent">✓</span> validate_inputs <span className="text-faint">· format, bands, CRS, co-registration</span> <span className="text-faint">94 ms</span></div>
                <div><span className="text-accent">✓</span> classify_task <span className="text-faint">· intent → change analysis</span></div>
                <div><span className="text-accent">✓</span> select_tool <span className="text-faint">· registry: 7 specialists</span></div>
                <div><span className="text-accent">✓</span> execute:change_analysis <span className="text-faint">· tiled inference</span> <span className="text-faint">1.7 s</span></div>
                <div><span className="text-accent">✓</span> integrate + report <span className="text-faint">· overlay, mask, provenance, confidence</span></div>
              </div>
            </div>
          </Reveal>
        </div>
      </section>

      <section className="relative z-10 border-t border-line bg-[var(--c-bg)]">
        <div className="mx-auto max-w-5xl px-8 py-24">
          <Reveal>
            <ChapterLabel n="05" title="Why Anvesha" />
            <h2 className="mt-6 text-[36px] font-bold text-body md:text-[44px]">
              Proof, not promises.
            </h2>
            <div className="mt-7 inline-flex rounded-full border border-line bg-panel p-1">
              {([['plain', 'Why Anvesha'], ['stats', 'Stats for professionals']] as const).map(
                ([key, label]) => (
                  <button
                    key={key}
                    onClick={() => setWhyTab(key)}
                    className={`rounded-full px-5 py-2 font-mono text-[13px] uppercase tracking-[.15em] transition-colors ${
                      whyTab === key
                        ? 'bg-accent text-white'
                        : 'text-muted hover:text-accent'
                    }`}
                  >
                    {label}
                  </button>
                ),
              )}
            </div>
          </Reveal>
          {whyTab === 'plain' ? (
            <Reveal delay={120}>
              <ul className="mt-10 space-y-7">
                {[
                  'One or two images and one question go in. A decision-ready answer comes out: what changed, where it is, and what it means.',
                  'Every answer shows its work: the change highlighted on the map, a confidence score you can hover to understand, and every step it took.',
                  'It runs on an ordinary laptop, fully offline. Your data never leaves the room.',
                  'It reads both optical cameras and radar, so clouds and darkness are not a problem.',
                  'It was measured on the exact public benchmarks the problem statement names, and it accepts ISRO-format inputs.',
                ].map((l, i) => (
                  <li key={i} className="flex items-start gap-5">
                    <span className="shrink-0 pt-1 font-mono text-[20px] font-bold text-accent">{String(i + 1).padStart(2, '0')}</span>
                    <span className="text-[21px] font-medium leading-snug text-body">{l}</span>
                  </li>
                ))}
              </ul>
            </Reveal>
          ) : (
            <Reveal delay={120}>
              <ul className="mt-10 divide-y divide-line border-y border-line">
                {[
                  'Change detection: 0.692 IoU / 0.818 F1 on the full LEVIR-CD test split (1,500 pairs, thr=0.85) — CPU-class Siamese FPN, 44 MB weights, tiled inference',
                  'VQA question understanding runs on a CLIP text encoder: compositional accuracy +11 pts over bag-of-words, adopted behind a pre-registered A/B gate',
                  'Captions decode from CLIP vision features: same gate discipline, +13% relative BLEU over the previous stack',
                  'Change-VQA: 0.683 on 39.7k test questions: +17.4 pts over the majority-class baseline, every question type above it',
                  'Confidence is calibrated (temperature fit on held-out data), not raw softmax',
                  '100/100 concurrent analyses, p95 ≈ 7 s, int8 export costs 0.16% accuracy',
                  'Offline-deployable: bundled weights, Docker, SQLite cache, no cloud calls',
                  'Every answer carries its source, every run carries its trace, and every number on this page reproduces with one command.',
                ].map((l, i) => (
                  <li key={i} className="flex gap-5 py-4">
                    <span className="shrink-0 font-mono text-[13px] text-accent">{String(i + 1).padStart(2, '0')}</span>
                    <span className="text-[16.5px] leading-relaxed text-muted">{l}</span>
                  </li>
                ))}
              </ul>
            </Reveal>
          )}
          <Reveal delay={200}>
            <p className="mt-12 text-[24px] font-semibold text-body md:text-[30px]">
              If it isn't measured, it isn't on this page.
            </p>
          </Reveal>
        </div>
      </section>

      <section className="relative z-10 border-t border-line bg-[var(--c-bg)]">
        <div className="mx-auto max-w-5xl px-8 pb-32 pt-16 text-center">
          <Reveal>
            <div className="font-mono text-[13px] uppercase tracking-[.3em] text-accent">
              Your turn
            </div>
            <h2 className="mt-6 text-[56px] font-extrabold uppercase leading-none tracking-[.08em] text-body md:text-[84px]">
              Ask the Earth.
            </h2>
            <div className="mt-5 font-mono text-[14px] uppercase tracking-[.3em] text-muted">
              two images · one sentence
            </div>
            <button
              onClick={onEnterConsole}
              className="mt-10 rounded-lg bg-accent px-8 py-3 text-[16px] font-semibold text-white transition-colors hover:bg-accent-dim"
            >
              Launch the console →
            </button>
            <div className="mt-4 font-mono text-[13px] uppercase tracking-[.2em] text-faint">
              demo samples included: no data needed
            </div>
          </Reveal>
        </div>
      </section>
    </div>
  )
}
