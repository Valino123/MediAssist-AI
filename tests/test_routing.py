import os
import types

import pytest


# Ensure the package path is available when running from repo root
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from agents.multi_agent_orchestrator import MultiAgentOrchestrator
from config import config


DISCLAIMER_SNIPPET = "This information is for educational purposes"


@pytest.fixture()
def orchestrator():
    return MultiAgentOrchestrator()


def test_conversation_route_goes_to_guardrails(monkeypatch, orchestrator):
    # Route decision → conversation
    monkeypatch.setattr(
        orchestrator.agent_decision_system,
        "decide_agent",
        lambda query, has_image=False, image_type=None: {
            "agent": "CONVERSATION_AGENT",
            "reasoning": "greeting",
            "confidence": 0.95,
        },
    )

    # Stub conversation agent
    from agents.simple_chat_agent import chat_agent

    monkeypatch.setattr(
        chat_agent,
        "process_message",
        lambda message: {"response": "Hello!"},
    )

    result = orchestrator.process_query("Hi", vector_store=None)

    assert result["agent"] == "CONVERSATION_AGENT"
    assert DISCLAIMER_SNIPPET in result["response"]


def test_rag_high_confidence_flows_to_guardrails(monkeypatch, orchestrator):
    # Decision → RAG
    monkeypatch.setattr(
        orchestrator.agent_decision_system,
        "decide_agent",
        lambda query, has_image=False, image_type=None: {
            "agent": "RAG_AGENT",
            "reasoning": "kb question",
            "confidence": 0.9,
        },
    )

    # Configure threshold and stub RAG
    monkeypatch.setattr(config, "RAG_CONFIDENCE_THRESHOLD", 0.7, raising=False)

    from agents.rag_agent import rag_agent

    monkeypatch.setattr(
        rag_agent,
        "process_query",
        lambda q: {"response": "KB answer", "confidence": 0.9},
    )

    result = orchestrator.process_query("What is hypertension?", vector_store=None)

    assert result["agent"] == "RAG_AGENT"
    assert DISCLAIMER_SNIPPET in result["response"]


def test_rag_low_confidence_routes_to_web_search(monkeypatch, orchestrator):
    # Decision → RAG
    monkeypatch.setattr(
        orchestrator.agent_decision_system,
        "decide_agent",
        lambda query, has_image=False, image_type=None: {
            "agent": "RAG_AGENT",
            "reasoning": "news-like question",
            "confidence": 0.6,
        },
    )

    # Low confidence relative to threshold
    monkeypatch.setattr(config, "RAG_CONFIDENCE_THRESHOLD", 0.8, raising=False)

    from agents.rag_agent import rag_agent

    monkeypatch.setattr(
        rag_agent,
        "process_query",
        lambda q: {"response": "Not enough KB info", "confidence": 0.5},
    )

    result = orchestrator.process_query("Is there a new outbreak today?", vector_store=None)

    assert result["agent"] == "WEB_SEARCH_AGENT"
    assert DISCLAIMER_SNIPPET in result["response"]
    assert "web search results" in result["response"].lower()


def test_image_routes_to_specific_agent(monkeypatch, orchestrator, tmp_path):
    # Decide to route to brain tumor agent
    monkeypatch.setattr(
        orchestrator.agent_decision_system,
        "decide_medical_image_agent",
        lambda query, image_path: {
            "agent": "BRAIN_TUMOR_AGENT",
            "reasoning": "mri brain",
            "confidence": 0.92,
            "image_type": "brain_mri",
        },
    )

    # Stub the agent's process_image method
    monkeypatch.setattr(
        orchestrator.brain_tumor_agent,
        "process_image",
        lambda data_uri: {"analysis_report": "Brain MRI analysis"},
    )

    # Create a dummy image file path
    dummy = tmp_path / "img.jpg"
    dummy.write_bytes(b"test")

    result = orchestrator.process_image_query("Analyze this", str(dummy), vector_store=None)

    assert result["agent"] == "BRAIN_TUMOR_AGENT"
    assert DISCLAIMER_SNIPPET in result["response"]


