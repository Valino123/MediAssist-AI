import base64
from pathlib import Path

import pytest
import torch

from agents.image_analysis_agent.brain_tumor_agent import BrainTumorAgent
from agents.image_analysis_agent.chest_xray_agent import ChestXrayAgent
from agents.image_analysis_agent.skin_lesion_agent import SkinLesionAgent

DATA_DIR = Path(__file__).parent / "data" / "image"


class _DummyModel:
    def __init__(self, labels):
        self.config = type("Cfg", (), {"id2label": labels})

    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self


@pytest.mark.parametrize(
    "agent_cls, filename, agent_key",
    [
        (BrainTumorAgent, "brain_tumor_01.jpg", "BRAIN_TUMOR_AGENT"),
        (ChestXrayAgent, "chest_xray_01.jpg", "CHEST_XRAY_AGENT"),
        (SkinLesionAgent, "skin_cancer_01.jpg", "SKIN_LESION_AGENT"),
    ],
)
def test_medical_image_agents(monkeypatch, agent_cls, filename, agent_key):
    """Each agent should process an image and return a structured result."""
    print(f"[test_image] start agent={agent_key} file={filename}")

    # Keep init lightweight: skip remote model loading
    def fake_load(self):
        self.model = _DummyModel({0: "normal", 1: "finding"})
        self.processor = object()

    # Fixed logits to avoid heavy inference
    def fake_predict(self, _image):
        return torch.tensor([[0.1, 0.9]])

    monkeypatch.setattr(agent_cls, "load_model", fake_load)
    monkeypatch.setattr(agent_cls, "predict", fake_predict)

    agent = agent_cls()

    img_path = DATA_DIR / filename
    encoded = base64.b64encode(img_path.read_bytes()).decode()
    result = agent.process_image(encoded)

    assert result["status"] == "success"
    assert agent_key in result["agent"]
    assert result["confidence"] >= 0
    assert result["analysis_report"]
