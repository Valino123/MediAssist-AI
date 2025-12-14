from types import SimpleNamespace
from pathlib import Path

import pytest

from agents import multi_agent_orchestrator as mao
from agents.agent_decision import AgentState


@pytest.fixture
def orchestrator(monkeypatch):
    """Lightweight orchestrator with heavy deps mocked out."""

    class DummyDecisionSystem:
        def decide_agent(self, *_args, **_kwargs):
            return {"agent": "RAG_AGENT", "reasoning": "kb", "confidence": 0.9}

        def decide_medical_image_agent(self, *_args, **_kwargs):
            return {"agent": "BRAIN_TUMOR_AGENT", "reasoning": "mri", "confidence": 0.8, "image_type": "brain"}

    class DummyImgAgent:
        def __init__(self, *args, **kwargs):
            pass

        def process_image(self, *_args, **_kwargs):
            # Include the keyword 'analysis' so routing tests can assert on it
            return {"analysis_report": "analysis ok", "status": "success"}

    monkeypatch.setattr(mao, "AgentDecisionSystem", lambda: DummyDecisionSystem())
    monkeypatch.setattr(mao, "_BTClass", DummyImgAgent)
    monkeypatch.setattr(mao, "_CXClass", DummyImgAgent)
    monkeypatch.setattr(mao, "_SLClass", DummyImgAgent)
    monkeypatch.setattr(mao.rag_agent, "process_query", lambda *_args, **_kwargs: {"response": "kb", "confidence": 0.95})
    monkeypatch.setattr(mao.chat_agent, "process_message", lambda *_args, **_kwargs: {"response": "hi"})

    class DummyWeb:
        def search_medical_info(self, *_args, **_kwargs):
            return {"status": "success", "results": {"summary": "web"}, "timestamp": "t"}

    monkeypatch.setattr(mao, "WebSearchAgent", lambda: DummyWeb())

    orch = mao.MultiAgentOrchestrator()
    return orch


def test_routing_rag(monkeypatch, orchestrator):
    print("[test_routing] routing_rag")
    state = AgentState(current_input="what is hypertension")
    orchestrator._run_rag_agent(state)
    assert state.selected_agent == "RAG_AGENT"
    assert "kb" in state.agent_response
    assert state.rag_confidence == pytest.approx(0.95)


def test_routing_web_search(monkeypatch, orchestrator):
    print("[test_routing] routing_web_search")
    state = AgentState(current_input="latest outbreak")
    orchestrator._run_web_search_agent(state)
    assert state.selected_agent == "WEB_SEARCH_AGENT"
    assert "web" in state.agent_response


def test_routing_image_brain(monkeypatch, orchestrator):
    print("[test_routing] routing_image_brain")
    img_path = Path(__file__).parent / "data" / "image" / "brain_tumor_01.jpg"
    state = AgentState(current_input="analyze brain", has_image=True, image_path=str(img_path))
    orchestrator._run_brain_tumor_agent(state)
    assert state.selected_agent == "BRAIN_TUMOR_AGENT"
    assert "analysis" in state.agent_response
