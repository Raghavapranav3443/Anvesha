"""Exception hierarchy for the online-acquisition layer.

``NetworkBlockedAirgap`` deliberately inherits from **both** ``AcquireError``
and ``OSError``:

* ``OSError`` so that library code which already treats an unreachable network
  as a recoverable condition degrades exactly as it would offline. This is what
  lets e.g. ``models/dino_encoder.py`` fall through to its bundled checkpoint
  instead of crashing model construction when the audit hook blocks a
  ``torch.hub.load`` fetch.
* ``AcquireError`` so acquisition code can catch air-gap refusals specifically
  without swallowing genuine network faults.

How a blocked call actually surfaces, layer by layer (verified by test):

* raw sockets -> ``NetworkBlockedAirgap`` directly (``socket.create_connection``,
  ``sendto``, ``getaddrinfo``);
* ``urllib`` -> ``urllib.error.URLError`` **wrapping** our exception, because
  ``do_open`` catches ``OSError`` and re-raises ``URLError(err)``. Inspect
  ``URLError.reason`` to recover it. Graceful degradation is preserved either
  way since ``URLError`` is itself an ``OSError``.

So: catch ``OSError`` (or ``URLError`` for urllib paths) to degrade, and check
``.reason`` / catch the class directly to distinguish an air-gap refusal from a
genuine network fault.
"""
from __future__ import annotations


class AcquireError(RuntimeError):
    """Base class for acquisition failures (never raised for air-gap).

    Air-gap refusals use :class:`NetworkBlockedAirgap`, which is an
    ``AcquireError`` *and* an ``OSError``; see the module docstring.
    """


class NetworkBlockedAirgap(AcquireError, OSError):
    """Raised when air-gap policy blocks an outbound network operation.

    Carries the audit event and target so the UI can show *what* was blocked,
    which is the whole point of an enforceable air gap.
    """

    def __init__(self, message: str, *, event: str = "", target: str = "") -> None:
        super().__init__(message)
        self.event = event
        self.target = target

    def __str__(self) -> str:  # keep the message stable for tests/log parsing
        return self.args[0] if self.args else "air-gap mode: outbound network blocked"


class NoCatalogAvailable(AcquireError):
    """No configured imagery catalogue is usable (missing credentials/config)."""


class NoSceneFound(AcquireError):
    """The catalogue returned no scene matching the request."""


class ScopeTooLarge(AcquireError):
    """The requested area is too large for a windowed (non-bulk) fetch."""


class FetchFailed(AcquireError):
    """A window/asset download failed after retries."""
