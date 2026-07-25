"""Backward-compatible entry point — delegates to the unified orchestrator."""

from src.agent.orchestrator import answer_query

__all__ = ["answer_query"]
