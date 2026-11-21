"""Scorers (Phase 5).

``execution_accuracy(task, result) -> Score``: canonicalize both result sets and compare
as multisets (or as sequences when ``task.ordered``). Optional ``llm_judge`` is off by
default: subjective and cost-bearing.
"""
