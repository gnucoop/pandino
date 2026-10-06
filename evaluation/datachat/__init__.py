"""DataChat Functional Evaluation harness.

Runs the frozen DataChat scenario suite against a Maui process it owns and
persists raw evidence for later, separate grading. It imports nothing from
the Maui application (no config, no database module, no app globals).
"""
