"""Source-specific evidence readers of the DataChat Functional Evaluation harness.

Each collector inspects one evidence source and returns an
``EvidenceSourceRecord`` (or artifact metadata) plus the raw payload to
persist. Collectors never write scenario files, never decide scenario or
suite status and never grade.
"""
