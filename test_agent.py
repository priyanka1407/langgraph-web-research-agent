"""Offline tests (mock mode). Run: python test_agent.py  or  pytest"""

import os

os.environ["MOCK_LLM"] = "1"

import agent  # noqa: E402


def base(**overrides):
    state = agent.initial_state("test query", max_searches=3)
    state.update(overrides)
    return state


def test_router_goes_to_synthesize_when_sufficient():
    assert agent.route_after_grade(base(is_sufficient=True, search_count=1)) == "synthesize"


def test_router_loops_when_not_sufficient_and_budget_left():
    assert agent.route_after_grade(base(is_sufficient=False, search_count=1)) == "search"


def test_router_stops_when_budget_used_up():
    assert agent.route_after_grade(base(is_sufficient=False, search_count=3)) == "synthesize"


def test_full_run_loops_once_then_reports():
    result = agent.build_graph().invoke(agent.initial_state("AI agents", max_searches=3))
    assert result["search_count"] == 2                      # searched, graded "no", searched again
    assert result["queries_tried"][1] != result["queries_tried"][0]   # second search used a refined query
    assert len(result["search_results"]) == 6               # operator.add kept both searches' results
    assert "## Summary" in result["report"]


def test_budget_of_one_skips_the_loop():
    result = agent.build_graph().invoke(agent.initial_state("AI agents", max_searches=1))
    assert result["search_count"] == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
