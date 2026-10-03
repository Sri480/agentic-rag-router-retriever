from crewai import Agent, Task, Crew
from crewai.tools import BaseTool

from config import get_llm
from models import RouteDecision


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class AgentExecutionError(Exception):
    """Raised when a Router or Retriever crew fails to produce output."""
    def __init__(self, stage: str, original: Exception):
        self.stage = stage
        self.original = original
        super().__init__(f"{stage} failed: {original}")

# ---------------------------------------------------------------------------
# Router Agent: decides RAG / WEB / BOTH. Has no tools; it only classifies.
# ---------------------------------------------------------------------------
def get_router_agent() -> Agent:
    return Agent(
        role="Query Router",
        goal=(
            "Analyze an incoming question and decide whether it should be "
            "answered from the internal PDF knowledge base, a live web "
            "search, or both."
        ),
        backstory=(
            "An experienced dispatcher who knows whether an answer lives in "
            "a static internal document or needs fresh, real-time information."
        ),
        llm=get_llm(),
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Retriever Agent: built per call with exactly ONE tool attached, so it can
# only consult the source the Router chose.
# ---------------------------------------------------------------------------
def get_retriever_agent(tool: BaseTool) -> Agent:
    return Agent(
        role="Retriever",
        goal=(
            f"Use the '{tool.name}' tool to find grounded evidence and answer "
            "the question, citing exactly what the tool returned."
        ),
        backstory=(
            "A careful research assistant who never answers from memory "
            "alone. It always checks its assigned source first."
        ),
        tools=[tool],
        llm=get_llm(),
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Helpers: each agent's work runs inside its own single-task Crew.
# ---------------------------------------------------------------------------
def run_router(query: str) -> RouteDecision:
    agent = get_router_agent()
    task = Task(
        description=(
            f"Decide how to answer this question: '{query}'. "
            "Choose RAG if it's likely covered by the internal PDF documents "
#            "(a research paper on transformers). "
            "Choose WEB if it requires current events or live data. "
            "Choose BOTH if a good answer combines the internal document "
            "with current/live context."
        ),
        expected_output="A routing decision: source ('RAG', 'WEB' or 'BOTH') and reasoning.",
        agent=agent,
        output_pydantic=RouteDecision,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    try:
        result = crew.kickoff()
    except Exception as e:
        raise AgentExecutionError("Router", e) from e

    if result.pydantic is None:
        # The LLM ran but didn't return valid structured output.
        raise AgentExecutionError("Router", ValueError("No structured RouteDecision returned"))

    return result.pydantic


def run_retriever(query: str, tool: BaseTool) -> str:
    agent = get_retriever_agent(tool)
    task = Task(
        description=(
            f"Answer this question using your assigned tool: '{query}'. "
            "Base your answer only on what the tool returns, and mention "
            "the source (file/page or URL) for each claim."
        ),
        expected_output="A concise, grounded answer that cites its sources.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    try:
        result = crew.kickoff()
    except Exception as e:
        raise AgentExecutionError(f"Retriever ({tool.name})", e) from e

    return result.raw

