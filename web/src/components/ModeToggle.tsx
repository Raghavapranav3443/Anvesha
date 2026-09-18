import { useCallback, useEffect, useRef, useState } from 'react'
import {
  fetchBlocked, fetchMode, setMode,
  type BlockedAttempt, type ModeState,
} from '../api'

/**
 * Air-gap ⇄ online mode control.
 *
 * Design rules this component follows deliberately:
 *
 * - The state shown is the *server's*. We never infer connectivity on the
 *   client, because a UI that guesses "online" while the server is refusing
 *   egress would be actively misleading.
 * - Air-gap is not a marketing label here: when it is on, the backend has a
 *   permanent audit hook refusing outbound connections, and this control can
 *   show you exactly what it refused.
 * - Switching to online does not weaken the guarantee. The guard stays
 *   installed and its policy flips, so switching back is enforced instantly.
 */
export function ModeToggle() {
  const [state, setState] = useState<ModeState | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [blocked, setBlocked] = useState<BlockedAttempt[]>([])
  const [showBlocked, setShowBlocked] = useState(false)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    fetchMode()
      .then(m => { if (mounted.current) setState(m) })
      .catch(() => { /* header must render even if the API is down */ })
    return () => { mounted.current = false }
  }, [])

  const toggle = useCallback(async () => {
    if (!state || busy) return
    const next = state.mode === 'airgap' ? 'online' : 'airgap'
    setBusy(true)
    setError('')
    try {
      const updated = await setMode(next)
      if (!mounted.current) return
      setState(updated)
      setShowBlocked(false)
      // Only air-gap mode can have something to show, so only poll then.
      if (updated.mode === 'airgap') {
        const b = await fetchBlocked(true)
        if (mounted.current) {
          setBlocked(b.blocked ?? [])
          setShowBlocked((b.blocked ?? []).length > 0)
        }
      } else {
        setBlocked([])
      }
    } catch (err) {
      if (mounted.current) {
        setError(err instanceof Error ? err.message : 'Could not switch mode.')
      }
    } finally {
      if (mounted.current) setBusy(false)
    }
  }, [state, busy])

  if (!state) {
    return (
      <span className="hidden items-center gap-1.5 rounded-full border border-line px-3 py-1 text-sm text-faint md:flex">
        <span className="h-2 w-2 rounded-full bg-faint/50" />
        checking mode…
      </span>
    )
  }

  const airgap = state.mode === 'airgap'
  const honest = state.guard_disabled_by_env
    ? { dot: 'bg-warn', text: 'air-gap guard disabled' }
    : airgap
      ? { dot: 'bg-sky-500', text: 'Air-gapped' }
      : { dot: 'bg-good', text: 'Online' }

  const hint = state.guard_disabled_by_env
    ? 'ANVESHA_AIRGAP_GUARD=0 is set, so outbound network is NOT being blocked.'
    : airgap
      ? 'Outbound network is blocked in this process. Click to allow fetching imagery.'
      : 'Outbound network is allowed. Click to return to air-gapped operation.'

  return (
    <span className="relative hidden items-center gap-2 md:flex">
      <button
        type="button"
        onClick={toggle}
        disabled={busy}
        title={hint}
        aria-pressed={!airgap}
        aria-label={`Operating mode: ${honest.text}. ${hint}`}
        className="flex items-center gap-1.5 rounded-full border border-line px-3 py-1 text-sm text-muted transition-colors hover:border-accent/50 hover:text-body disabled:opacity-60"
      >
        <span className={`h-2 w-2 rounded-full ${honest.dot}`} />
        {busy ? 'switching…' : honest.text}
        <span className="text-faint" aria-hidden>
          {airgap ? '· offline' : '· may fetch'}
        </span>
      </button>

      {blocked.length > 0 && (
        <button
          type="button"
          onClick={() => setShowBlocked(s => !s)}
          title="Shows what the air-gap guard blocked, for transparency"
          className="rounded-full border border-line px-2 py-1 font-mono text-xs text-faint hover:text-accent"
        >
          {blocked.length} blocked
        </button>
      )}

      {error && (
        <span className="absolute right-0 top-full mt-1 whitespace-nowrap rounded border border-warn/40 bg-panel px-2 py-1 text-xs text-warn shadow-[var(--shadow-panel)]">
          {error}
        </span>
      )}

      {showBlocked && blocked.length > 0 && (
        <div className="absolute right-0 top-full z-40 mt-2 w-80 rounded-xl border border-line bg-panel p-3 text-left shadow-[var(--shadow-panel)]">
          <div className="mb-2 text-xs font-semibold uppercase tracking-[.13em] text-faint">
            Blocked outbound attempts
          </div>
          <ul className="space-y-1">
            {blocked.slice(-5).reverse().map((b, i) => (
              <li key={`${b.at}-${i}`} className="font-mono text-xs text-muted">
                <span className="text-body">{b.event}</span> → {b.target}
                {b.where ? <span className="text-faint"> ({b.where})</span> : null}
              </li>
            ))}
          </ul>
          <p className="mt-2 text-xs text-faint">
            Nothing left this machine. This is the guarantee, not a claim about it.
          </p>
        </div>
      )}
    </span>
  )
}

export default ModeToggle
