
# ---------------------------------------------------------------------------
# import required libraries
# ---------------------------------------------------------------------------
from crewai.flow.flow import Flow, start, listen, router, or_

from models import RagFlowState
from agents import run_router, run_retriever
from tools.pdf_tool import get_pdf_tool
from tools.web_tool import get_web_tool
from logging_utils import trace_logger


# ---------------------------------------------------------------------------
# Built once, reused across every run: PDFSearchTool indexes the PDF on
# construction (embedding calls), so we don't want to rebuild it per query.
# ---------------------------------------------------------------------------
PDF_TOOL = get_pdf_tool("sample_docs/trasformer_research_paper_dataset.pdf")
WEB_TOOL = get_web_tool()


# ---------------------------------------------------------------------------
# Flow class for this application with @router and @listen decorators.
# This is where the architecture of the flow is implemented. 
# ---------------------------------------------------------------------------
class RagFlow(Flow[RagFlowState]):

    @start()
    def receive_query(self):
        self.run_id = self.state.query[:30]  # cheap unique-enough id for this run
        trace_logger.log(self.run_id, "Flow", "receive_query", self.state.query)

    @router(receive_query)
    def route(self):
        try:
            decision = run_router(self.state.query)
        except AgentExecutionError as e:
            # Router itself failed -- fall back to WEB rather than crashing,
            # since a web answer beats no answer at all.
            trace_logger.log(self.run_id, "Router Agent", "ERROR", str(e))
            self.state.route = "WEB"
            self.state.reasoning = f"Routing failed ({e}); defaulted to WEB."
            return "WEB"

        self.state.route = decision.source
        self.state.reasoning = decision.reasoning
        trace_logger.log(self.run_id, "Router Agent", "route", f"{decision.source} -- {decision.reasoning}")
        return decision.source  # drives which @listen branch below fires

    @listen("RAG")
    def retrieve_pdf(self):
        try:
            self.state.pdf_context = run_retriever(self.state.query, PDF_TOOL)
            trace_logger.log(self.run_id, "Retriever Agent (PDF)", "retrieve", self.state.pdf_context)
        except AgentExecutionError as e:
            trace_logger.log(self.run_id, "Retriever Agent (PDF)", "ERROR", str(e))
            self.state.pdf_context = f"[PDF retrieval failed: {e}]"

    @listen("WEB")
    def retrieve_web(self):
        try:
            self.state.web_context = run_retriever(self.state.query, WEB_TOOL)
            trace_logger.log(self.run_id, "Retriever Agent (WEB)", "retrieve", self.state.web_context)
        except AgentExecutionError as e:
            trace_logger.log(self.run_id, "Retriever Agent (WEB)", "ERROR", str(e))
            self.state.web_context = f"[Web retrieval failed: {e}]"

    @listen("BOTH")
    def retrieve_both(self):
        self.retrieve_pdf()
        self.retrieve_web()

    @listen(or_(retrieve_pdf, retrieve_web, retrieve_both))
    def synthesize(self):
        parts = [p for p in [self.state.pdf_context, self.state.web_context] if p]
        self.state.final_answer = "\n---\n".join(parts)
        trace_logger.log(self.run_id, "Flow", "synthesize", self.state.final_answer)

