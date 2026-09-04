"""Usage subsystem — token and cost accounting for Maui.

This package is the physical boundary of the Usage subsystem. It performs no
imports of its own: every stable surface is an explicit submodule, so
importing one Usage capability never drags in the others (notably, the
provider-facing accounting modules stay free of Flask and of the database
layer, which the capture wrapper in ``infrastructure/embedding_capture.py``
depends on because it runs on a background event-loop thread).

Stable surfaces
---------------

Adopter surface — routes and services:

    usage.recording                 record_token_consumption,
                                    record_resolved_consumption
    usage.attribution               attribute_usage_to_user,
                                    attribute_usage_to_policy,
                                    declare_usage_unattributed,
                                    USAGE_POLICY_*
    usage.embedding_operation_context
                                    embedding_operation, OPERATION_QUERY,
                                    OPERATION_DOCUMENT, OPERATION_PROBE

Composition-root surface — ``main.py`` only:

    usage.lifecycle                 register_usage_lifecycle_hooks

Provider-producer surface — the embedding capture adapter only:

    usage.embedding_accounting      EmbeddingAccountingContribution and the
                                    quantity/origin/cost-state vocabulary
    usage.embedding_accounting_sink get_embedding_accounting_sink
    usage.embedding_operation_context
                                    get_embedding_operation

Compatibility surface — routes whose existing client contract returns
``log_id``:

    usage.request_state             get_usage_log_id

Everything else in this package is implementation detail: the request-state
binders, the aggregation and provenance vocabularies, the persistence and
finalization hooks, and the individual lifecycle registrars that
``usage.lifecycle`` composes.

Dependencies Usage has but does not own
---------------------------------------

    utils.request_duration          request timing; the lifecycle boundary
                                    orders its registrar but does not own
                                    timing semantics
    utils.logging_config            request context / ``request_id``
    infrastructure.database_pg      persistence

Operational Persistence is a separate subsystem and is not a dependency of
this package.
"""
