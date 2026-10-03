# ---------------------------------------------------------------------------
# This module sets up the necessary models for interactive with CrewAI
# ---------------------------------------------------------------------------
from typing import Literal
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Structured output the Router Agent must produce."""
    source: Literal["RAG", "WEB", "BOTH"]
    reasoning: str


class RagFlowState(BaseModel):
    """Shared state passed between every step of the Flow."""
    query: str = ""
    route: str = ""
    reasoning: str = ""
    pdf_context: str = ""
    web_context: str = ""
    final_answer: str = ""

