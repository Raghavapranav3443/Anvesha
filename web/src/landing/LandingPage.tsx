import { useEffect, useRef, useState } from 'react'
import HeroGlobe from './HeroGlobe'
import ManifestTable from './ManifestTable'
import PipelineTrack from './PipelineTrack'
import Reveal from './Reveal'
import Ticker from './Ticker'
import './landing.css'

function ChapterLabel({ n, title }: { n: string; title: string }) {
  return (
    <div className="flex items-center gap-3 font-mono text-[13px] uppercase tracking-[.3em]">
      <span className="text-accent">{n}</span>
      <span className="h-px w-10 bg-accent/50" />
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

  const zoom = reduced ? 0 : Math.min(progress / 0.6, 1)
  const fade = reduced ? 0 : Math.min(1, Math.max(0, progress - 0.18) / 0.3)

  return (
    <div className="relative bg-black">
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
          className="rounded-lg border border-transparent bg-panel/80 px-3 py-1.5 text-[15px] text-body backdrop-blur transition-colors hover:border-accent/50"
        >
          ☀ / ☾
        </button>
      </div>

      {/* scroll track: hero stays pinned while the camera dollies in */}
      <div ref={trackRef} className="relative" style={{ height: '220vh' }}>
        <div className="sticky top-0 h-screen overflow-hidden">
          <div className="absolute inset-0">
            {webgl ? <HeroGlobe zoom={zoom} reduced={reduced} /> : <StaticGlobeFallback />}
          </div>

          {/* centered overlay — name + tagline + CTA, fading as the globe grows */}
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
              <span className="rounded-full border border-white/20 bg-white/5 px-3 py-1 backdrop-blur">Cartosat-2S + RISAT ready</span>
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
        <div className="mx-auto max-w-5xl px-8 py-28">
          <Reveal>
            <ChapterLabel n="01" title="The problem" />
            <h2 className="mt-6 text-[36px] font-bold leading-tight text-body md:text-[44px]">
              The data arrived.<br />The answer didn't.
            </h2>
          </Reveal>
          <Reveal delay={120}>
            <ul className="mt-10 space-y-3 font-mono text-[14.5px] leading-relaxed text-muted">
              <li className="flex gap-3"><span className="text-accent">▸</span>every satellite pass — hundreds of km² of pixels</li>
              <li className="flex gap-3"><span className="text-accent">▸</span>change hides between two dates</li>
              <li className="flex gap-3"><span className="text-accent">▸</span>finding it is specialist work: GIS tools, manual digitising, hours per pair</li>
            </ul>
          </Reveal>
          <Reveal delay={200}>
            <div className="mt-12 space-y-2 font-mono">
              <div className="text-[16.5px] uppercase tracking-[.18em] text-faint line-through decoration-faint">
                1 question → days of tooling
              </div>
              <div className="text-[30px] font-bold uppercase tracking-[.12em] text-body md:text-[38px]">
                1 sentence → minutes
              </div>
            </div>
            <p className="mt-10 max-w-xl text-[17px] text-muted">
              Decision-makers don't need more pixels. They need answers.
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
              fine-tuned remote-sensing specialists, and returns evidence —
              every step timed and auditable.
            </p>
          </Reveal>
          <Reveal delay={150}><div className="mt-8"><PipelineTrack /></div></Reveal>
          <Reveal delay={250}>
            <div className="mt-6 font-mono text-[13px] uppercase tracking-[.25em] text-faint">
              investigation mode — change → water → impact → priority zones, chained automatically
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
              <div className="font-mono text-[12px] uppercase tracking-[.25em] text-faint">the scorecard — every row measured, one command</div>
              <div className="mt-4 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line md:grid-cols-3">
                {[
                  ['RSVQA-LR', '0.773', 'exact-match · n=282'],
                  ['LEVIR-CD', '0.724', 'change IoU · n=300'],
                  ['CDVQA', '0.646', 'answer-match · n=2,088'],
                  ['BigEarthNet captions', '0.306', 'multi-ref BLEU · n=300'],
                  ['VRSBench grounding', '0.126', 'IoU@0.5 · spectral · n=455'],
                  ['VRSBench captioning', '0.000', 'reported honestly — see below'],
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
            <div className="mt-6 font-mono text-[13px] uppercase tracking-[.2em] text-faint">
              measured on public test subsets · reproduce: python -m satquery.evaluate --all
            </div>
          </Reveal>
        </div>
      </section>

      <section className="relative z-10 border-t border-line bg-[var(--c-bg)]">
        <Ticker />
        <div className="mx-auto max-w-5xl px-8 py-24">
          <Reveal>
            <ChapterLabel n="04" title="Why Anvesha" />
            <h2 className="mt-6 text-[36px] font-bold text-body md:text-[44px]">
              Proof, not promises.
            </h2>
          </Reveal>
          <Reveal delay={120}>
            <ul className="mt-10 divide-y divide-line border-y border-line">
              {[
                'Change detection trained jointly on LEVIR-CD + SECOND (9.9k pairs): 0.72 IoU / 0.84 F1 — up from 0.67/0.80',
                'VQA question understanding runs on a CLIP text encoder — compositional accuracy +11 pts over bag-of-words, adopted behind a pre-registered A/B gate',
                'Captions decode from CLIP vision features — same gate discipline, +13% relative BLEU over the previous stack',
                'Change-VQA: 0.683 on 39.7k test questions — +17.4 pts over baseline, every question type above it',
                'Confidence is calibrated (temperature fit on held-out data), not raw softmax',
                '100/100 concurrent analyses · p95 ≈ 7 s · int8 export costs 0.16% accuracy',
                'Offline-deployable: bundled weights, Docker, SQLite cache — no cloud calls',
                'And the record of what we refused to ship: DINOv2 won the backbone bake-off but added zero VQA gain — rejected. Learned grounding heads peaked at 0.15 IoU — retired. CLIP failed our grounding gate twice — rejected with the numbers published. A VRSBench caption fine-tune was declined rather than half-done. Every answer still carries its source; every run still carries its trace.',
              ].map((l, i) => (
                <li key={i} className="flex gap-5 py-4">
                  <span className="shrink-0 font-mono text-[13px] text-accent">{String(i + 1).padStart(2, '0')}</span>
                  <span className="text-[16.5px] leading-relaxed text-muted">{l}</span>
                </li>
              ))}
            </ul>
          </Reveal>
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
              demo samples included — no data needed
            </div>
          </Reveal>
        </div>
      </section>
    </div>
  )
}
