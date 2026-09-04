# utils/usage_lifecycle.py
"""Runtime composition of the Usage subsystem's request-lifecycle hooks.

Owns exactly one responsibility: register, in the single order production
supports, the hook systems whose interleaving the Usage lifecycle depends
on. It registers no Flask hook of its own, holds no state, performs no
persistence, measures no time, reads no configuration and handles no
request - every effect below belongs to one of the modules it composes,
each of which remains public and individually usable.

Ownership and composition are different concerns.
``utils.request_duration`` is *not* a Usage module: it is shared request
timing infrastructure with no Usage knowledge, and it stays that way. Its
registrar is nevertheless called from here, because the Usage lifecycle
needs it interleaved at one exact point - after embedding Usage
persistence and before embedding accounting - and no caller can supply
that interleaving without re-acquiring the ordering knowledge this module
exists to own. Composing a prerequisite is not owning it: this module
contributes no timer implementation and no timing semantics.

Registration order is a correctness invariant, not style. Flask runs
``before_request`` in registration order (FIFO) and ``after_request`` and
``teardown_request`` in *reverse* registration order (LIFO), so one
sequence produces two chains that read in opposite directions::

    registered  1. Usage duration finalization
                2. embedding Usage persistence
                3. request duration timer
                4. embedding accounting lifecycle

    before_request  request timer start -> embedding sink bind
    after_request   request duration finalization
                        -> embedding Usage persistence
                        -> Usage duration finalization
    teardown        embedding sink unbind
                        -> embedding Usage persistence fallback

Each position is load-bearing. The duration finalizer is registered first
so it runs *last* in ``after_request``, by which point the duration is
finalized and embedding persistence has registered the Usage row ids it
must update. The timer is registered third so it starts *before* the
embedding sink binds - keeping the sink bind outside the interval
``duration_ms`` measures - and finalizes *first* in ``after_request``, so
the value is settled before any Usage reader looks at it. Embedding
accounting is registered last so its teardown runs *first*, unbinding the
capture sink before the persistence fallback; that is safe and deliberate,
because contributions live on the ``flask.g`` accumulator rather than
behind the sink and stay readable after unbinding.

One prerequisite remains external and belongs to the caller:
``utils.logging_config.register_request_context_hooks`` must be registered
*before* this function, so its ``before_request`` binds ``request_id``
first and its teardown unwinds last, leaving that id readable while the
persistence fallback writes.
"""

from utils.embedding_accounting_lifecycle import register_embedding_accounting_hooks
from utils.embedding_usage_persistence import (
    register_embedding_usage_persistence_hooks,
)
from utils.request_duration import register_request_duration_hooks
from utils.usage_duration_finalization import (
    register_usage_duration_finalization_hooks,
)

__all__ = ["register_usage_lifecycle_hooks"]


def register_usage_lifecycle_hooks(app) -> None:
    """Register the Usage lifecycle's hook systems, in the required order.

    Must be called before the app serves its first request, per the same
    Flask 2.3+ constraint documented on each registrar, and after
    :func:`utils.logging_config.register_request_context_hooks`.

    Idempotent by delegation, not by its own marker: every registrar
    called here already guards with a private app marker, so a second call
    registers nothing twice and this boundary introduces no further
    registration state to keep in sync.
    """
    register_usage_duration_finalization_hooks(app)
    register_embedding_usage_persistence_hooks(app)
    register_request_duration_hooks(app)
    register_embedding_accounting_hooks(app)
