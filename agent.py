"""
Web Research Agent with a conditional search loop (LangGraph + Tavily + OpenAI).

Flow:
    START -> search -> grade --(enough info?)-- yes --> synthesize -> END
               ^         |
               |___ no __|   (grader writes a better query; capped by max_searches)

Usage:
    python agent.py                                    # mock mode when API keys are missing
    python agent.py --query "latest advances in quantum computing"
    python agent.py --max-searches 2
    python agent.py --show-graph                       # print the graph as Mermaid
    MOCK_LLM=1 python agent.py                         # force mock mode (no keys, no cost)
"""

import argparse
import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MOCK = os.getenv("MOCK_LLM") == "1" or not (os.getenv("OPENAI_API_KEY") and os.getenv("TAVILY_API_KEY"))


# ---------------------------------------------------------------------------
# State: the shared "notebook" every node reads from and writes to
# ---------------------------------------------------------------------------
class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]               # chat history (appended)
    query: str                                             # original question, never changes
    current_query: str                                     # query for the NEXT search (grader may refine it)
    queries_tried: Annotated[list[str], operator.add]      # every query we searched (appended)
    search_results: Annotated[list[dict], operator.add]    # results from ALL searches (appended)
    search_count: int                                      # how many searches so far
    max_searches: int                                      # loop budget, stops infinite loops
    is_sufficient: bool                                    # grader's verdict
    grade_reason: str                                      # grader's one-line explanation
    report: str                                            # final output


class Grade(BaseModel):
    """Structured verdict from the grader LLM."""
    is_sufficient: bool = Field(description="True if the results cover the query well enough to write a report")
    reason: str = Field(description="One sentence explaining the verdict")
    refined_query: str = Field(description="A better web search query that fills the gaps; empty if sufficient")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tavily_search(query: str) -> list[dict]:
    from langchain_tavily import TavilySearch

    raw = TavilySearch(max_results=5).invoke(query)
    if isinstance(raw, dict):      # newer versions: {"results": [...]}
        return raw.get("results", [])
    if isinstance(raw, list):      # older versions: [...]
        return raw
    return []


def unique_results(results: list[dict]) -> list[dict]:
    """Two searches can return the same page; keep the first copy of each URL."""
    seen, out = set(), []
    for r in results:
        url = r.get("url")
        if url in seen:
            continue
        seen.add(url)
        out.append(r)
    return out


def format_results(results: list[dict]) -> str:
    return "\n\n".join(
        f"Source: {r.get('url', 'N/A')}\nTitle: {r.get('title', 'N/A')}\nContent: {r.get('content', '')[:500]}"
        for r in results
    )


def get_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=MODEL, temperature=0)


# ---------------------------------------------------------------------------
# Mock mode: same shapes as the real APIs, so the graph runs with no keys.
# The mock grader says "not enough" on the first pass, so you SEE the loop.
# ---------------------------------------------------------------------------
def mock_search(query: str, n: int) -> list[dict]:
    return [
        {"url": f"https://example.com/search{n}/article{i}", "title": f"[mock] {query} - article {i}",
         "content": f"Mock finding {i} from search {n} about '{query}'."}
        for i in range(1, 4)
    ]


def mock_grade(state: ResearchState) -> Grade:
    if state["search_count"] < 2:
        return Grade(is_sufficient=False, reason="Results are generic; no concrete examples or recent dates.",
                     refined_query=f"{state['query']} concrete examples 2026")
    return Grade(is_sufficient=True, reason="Results now include specific, recent examples.", refined_query="")


def mock_report(state: ResearchState, results: list[dict]) -> str:
    findings = "\n".join(f"- {r['content']}" for r in results)
    sources = "\n".join(f"- {r['url']}" for r in results)
    return (f"## Summary\n[mock] Report on '{state['query']}' built from {len(results)} unique sources "
            f"across {state['search_count']} searches.\n\n## Key Findings\n{findings}\n\n## Sources\n{sources}")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def search_web(state: ResearchState) -> dict:
    query = state["current_query"]
    n = state["search_count"] + 1
    results = mock_search(query, n) if MOCK else tavily_search(query)
    print(f"  [search #{n}] '{query}' -> {len(results)} results")
    return {"search_results": results, "queries_tried": [query], "search_count": n}


def grade_results(state: ResearchState) -> dict:
    """Ask the LLM: is what we found enough to answer the question?"""
    if MOCK:
        grade = mock_grade(state)
    else:
        grader = get_llm().with_structured_output(Grade)
        grade = grader.invoke([
            SystemMessage(content=(
                "You review web search results for a research analyst. Decide if they are enough to write "
                "a solid report (relevant, specific, reasonably recent, more than one source). If not, write "
                "ONE better search query that fills the biggest gap. Do not repeat a query already tried."
            )),
            HumanMessage(content=(
                f"Research query: {state['query']}\n"
                f"Queries already tried: {state['queries_tried']}\n\n"
                f"Search results:\n{format_results(unique_results(state['search_results']))}"
            )),
        ])

    print(f"  [grade]  sufficient={grade.is_sufficient} - {grade.reason}")
    update = {"is_sufficient": grade.is_sufficient, "grade_reason": grade.reason}
    if not grade.is_sufficient and grade.refined_query.strip():
        update["current_query"] = grade.refined_query.strip()
    return update


def route_after_grade(state: ResearchState) -> str:
    """The conditional edge: returns the NAME of the next node."""
    if state["is_sufficient"]:
        return "synthesize"
    if state["search_count"] >= state["max_searches"]:
        print("  [route]  search budget used up -> writing the report with what we have")
        return "synthesize"
    print(f"  [route]  not enough yet -> searching again with '{state['current_query']}'")
    return "search"


def synthesize_report(state: ResearchState) -> dict:
    results = unique_results(state["search_results"])
    if MOCK:
        return {"report": mock_report(state, results), "messages": [AIMessage(content="[mock report]")]}

    caveat = "" if state["is_sufficient"] else (
        f"\n\nNote: the reviewer judged these results incomplete ({state['grade_reason']}). "
        "Say clearly what could not be confirmed."
    )
    response = get_llm().invoke([
        SystemMessage(content="You are a research analyst. Synthesize the search results into a clear, structured "
                              "report with: Summary, Key Findings (bullet points), and Sources. Only use facts "
                              "from the search results."),
        HumanMessage(content=f"Research query: {state['query']}\n\nSearch results:\n{format_results(results)}{caveat}"),
    ])
    return {"report": response.content, "messages": [response]}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("search", search_web)
    graph.add_node("grade", grade_results)
    graph.add_node("synthesize", synthesize_report)

    graph.add_edge(START, "search")
    graph.add_edge("search", "grade")
    graph.add_conditional_edges(
        "grade",                                   # after this node...
        route_after_grade,                         # ...call this function to decide...
        {"search": "search", "synthesize": "synthesize"},   # ...and map its answer to a node
    )
    graph.add_edge("synthesize", END)
    return graph.compile()


def initial_state(query: str, max_searches: int) -> ResearchState:
    return {"messages": [], "query": query, "current_query": query, "queries_tried": [],
            "search_results": [], "search_count": 0, "max_searches": max_searches,
            "is_sufficient": False, "grade_reason": "", "report": ""}


def main():
    parser = argparse.ArgumentParser(description="Web Research Agent with a conditional search loop")
    parser.add_argument("--query", default="latest advances in AI agents", help="Research query")
    parser.add_argument("--max-searches", type=int, default=3, help="Maximum number of web searches")
    parser.add_argument("--show-graph", action="store_true", help="Print the graph as Mermaid and exit")
    args = parser.parse_args()

    agent = build_graph()
    if args.show_graph:
        print(agent.get_graph().draw_mermaid())
        return

    print(f"\nResearching: {args.query}  ({'MOCK mode' if MOCK else MODEL})\n")
    result = agent.invoke(initial_state(args.query, args.max_searches))

    print("\n" + "=" * 60)
    print(f"RESEARCH REPORT  (searches: {result['search_count']}, queries: {result['queries_tried']})")
    print("=" * 60)
    print(result["report"])


if __name__ == "__main__":
    main()
