# ---------------------------------------------------------------------------
# Import libraries
# ---------------------------------------------------------------------------
import argparse
import config  # (importing this loads .env and validates keys early)
from flow import RagFlow
from logging_utils import trace_logger
from results_logger import save_result

# ---------------------------------------------------------------------------
# Setting up the demo questions here to test the application.
# Each question should iterate over the entire flow and return data from PDF, WEB or BOTH.
# ---------------------------------------------------------------------------
DEMO_QUESTIONS = [
    "What is the attention mechanism in transformers?",
    "What is the latest news on Apple stock?",
    "How does the original transformer architecture compare to models released this year?",
]

# ---------------------------------------------------------------------------
# Run the flow for one question
# ---------------------------------------------------------------------------
def ask(query: str) -> dict:
    """Run one question through the full Router -> Retriever -> Synthesize flow."""
    flow = RagFlow()
    flow.kickoff(inputs={"query": query})
    return {
        "query": query,
        "route": flow.state.route,
        "reasoning": flow.state.reasoning,
        "answer": flow.state.final_answer,
        "run_id": flow.run_id,
    }


# ---------------------------------------------------------------------------
# Print Results
# ---------------------------------------------------------------------------
def print_result(result: dict) -> None:
    print("\n" + "=" * 70)
    print("Q:", result["query"])
    print("-" * 70)
    print(f"Route: {result['route']}  ({result['reasoning']})")
    print("-" * 70)
    print(result["answer"])
    print("-" * 70)
    print("Trace:")
    trace_logger.print_trace(result["run_id"])
    print("=" * 70)

# ---------------------------------------------------------------------------
# Main function to kick off the application
# For each question in DEMO_QUESTIONS, the flow is run for that particular question.
# Alternatively, the user can feed in a question at run time as argument using the
# -q option 
# e.g. python main.py -q "What is the attention mechanism in transformers?" 
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Agentic RAG: Router + Retriever demo")
    parser.add_argument(
        "-q", "--question",
        help="Ask a single question instead of running the demo set.",
    )
    args = parser.parse_args()

    questions = [args.question] if args.question else DEMO_QUESTIONS

    for q in questions:
        try:
            result = ask(q)
            print_result(result)
            save_result(result)
            trace_logger.save_to_file("outputs/trace_log.jsonl", run_id=result["run_id"])
        except Exception as e:
            print(f"\n[ERROR] Failed to process '{q}': {e}\n")


if __name__ == "__main__":
    main()

