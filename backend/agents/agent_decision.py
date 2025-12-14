import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import base64

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel 

from config import config, logger

class AgentDecision(BaseModel):
    """Output structure for agent decision"""
    agent: str
    reasoning: str
    confidence: float

class MedicalImageDecision(BaseModel):
    """Output structure for medical image agent decision"""
    agent: str
    reasoning: str
    confidence: float
    image_type: str

class AgentState(BaseModel):
    messages: List[Dict[str, str]] = []
    current_input: str = ""
    has_image: bool = False
    image_type: Optional[str] = None
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    requires_disclaimer: bool = False
    search_type: Optional[str] = None  # 'general' | 'literature' (PubMed) | None
    selected_agent: Optional[str] = None
    agent_response: Optional[str] = None
    rag_confidence: Optional[float] = None
    confidence: float = 0.0
    timestamp: str = ""

class AgentDecisionSystem:
    def __init__(self):
        self.decision_llm = ChatOpenAI(
            model=config.AGENT_DECISION_MODEL,
            temperature=float(config.AGENT_DECISION_TEMPERATURE or "0.1"),
            openai_api_key=config.DASHSCOPE_API_KEY,
            base_url=config.BASE_URL,
        )

        self.json_parser = JsonOutputParser(pydantic_object=AgentDecision)
        self.medical_image_parser = JsonOutputParser(pydantic_object=MedicalImageDecision)

        self.decision_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent medical triage system that routes user queries to the appropriate specialized agent.

Available agents:
1. CONVERSATION_AGENT - For general chat, greetings, and non-medical questions
2. RAG_AGENT - For specific medical knowledge questions that can be answered from established medical knowledge base
3. WEB_SEARCH_AGENT - For questions about recent medical developments, current outbreaks, or time-sensitive medical information
4. IMAGE_ANALYSIS_AGENT - For analysis of medical images (brain MRI, chest X-ray, skin lesions)

Make your decision based on these guidelines:
- If the user has not uploaded any image, route to conversation agent for general questions
- If the user asks about specific medical knowledge, use the RAG agent
- If the user asks about recent medical developments, use the web search agent
- If the user uploads a medical image, route to the IMAGE_ANALYSIS_AGENT (which will internally route to the appropriate specialized medical image agent)
- For general conversation, greetings, or non-medical questions, use the conversation agent

You must provide your answer in JSON format with the following structure:
{{
    "agent": "AGENT_NAME",
    "reasoning": "Your step-by-step reasoning for selecting this agent",
    "confidence": 0.95
}}"""),
            ("human", "User query: {query}\\nHas image: {has_image}\\nImage type: {image_type}")
        ])

        self.medical_image_decision_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical image analysis specialist that routes medical images to the appropriate specialized analysis agent.

Available medical image analysis agents:
1. BRAIN_TUMOR_AGENT - For brain MRI images, tumor detection, neurological conditions
2. CHEST_XRAY_AGENT - For chest X-ray images, lung conditions, pneumonia, respiratory issues
3. SKIN_LESION_AGENT - For skin lesion images, moles, rashes, dermatological conditions

Analyze the uploaded medical image and determine which specialized agent should handle it. Consider:
- The type of medical image (MRI, X-ray, skin photo, etc.)
- The anatomical region visible in the image
- Any visible medical conditions or abnormalities
- The user's question or context about the image

You must provide your answer in JSON format with the following structure:
{{
    "agent": "SPECIFIC_MEDICAL_AGENT_NAME",
    "reasoning": "Your detailed analysis of the image and reasoning for selecting this agent",
    "confidence": 0.95,
    "image_type": "brief description of the image type"
}}"""),
            ("human", [
                {"type": "text", "text": "User query: {query}"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_base64}"}}
            ])
        ])

        logger.info("AgentDecisionSystem initialized with multimodal support")

    # --- Helper normalization utilities to make routing resilient ---
    def _normalize_confidence(self, value: Any) -> float:
        try:
            v = float(value)
            if v < 0.0:
                return 0.0
            if v > 1.0:
                return 1.0
            return v
        except Exception:
            return 0.0

    def _normalize_agent(self, agent_name: str) -> str:
        if not agent_name:
            return "CONVERSATION_AGENT"
        name = str(agent_name).strip().upper().replace("-", "_").replace(" ", "_")

        # Common synonyms/abbreviations mapping
        synonyms = {
            "CHAT": "CONVERSATION_AGENT",
            "CONVERSATION": "CONVERSATION_AGENT",
            "LLM": "CONVERSATION_AGENT",
            "RAG": "RAG_AGENT",
            "RETRIEVAL": "RAG_AGENT",
            "SEARCH": "WEB_SEARCH_AGENT",
            "WEB": "WEB_SEARCH_AGENT",
            "BING": "WEB_SEARCH_AGENT",
            "GOOGLE": "WEB_SEARCH_AGENT",
            "IMAGE": "IMAGE_ANALYSIS_AGENT",
            "VISION": "IMAGE_ANALYSIS_AGENT",
        }

        if name in synonyms:
            return synonyms[name]

        allowed = {
            "CONVERSATION_AGENT",
            "RAG_AGENT",
            "WEB_SEARCH_AGENT",
            "IMAGE_ANALYSIS_AGENT",
            "BRAIN_TUMOR_AGENT",
            "CHEST_XRAY_AGENT",
            "SKIN_LESION_AGENT",
        }
        return name if name in allowed else "CONVERSATION_AGENT"

    def decide_agent(self, query: str, has_image: bool = False, image_type: str = None) -> Dict[str, Any]:
        """Decide which agent should handle the query"""
        try:
            decision_chain = self.decision_prompt | self.decision_llm | self.json_parser

            decision = decision_chain.invoke({
                "query": query,
                "has_image": has_image,
                "image_type": image_type or "None"
            })

            normalized_agent = self._normalize_agent(decision.get("agent"))
            normalized_conf = self._normalize_confidence(decision.get("confidence", 0.0))

            logger.info(f"Agent decision: {normalized_agent} (confidence: {normalized_conf})")

            return {
                "status": "success",
                "agent": normalized_agent,
                "reasoning": decision["reasoning"],
                "confidence": normalized_conf,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in agent decision: {str(e)}")
            return {
                "status": "error",
                "agent": "CONVERSATION_AGENT",  # Fallback
                "reasoning": "Error in decision making, falling back to conversation agent",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def decide_medical_image_agent(self, query: str, image_path: str) -> Dict[str, Any]:
        """Decide which specific medical image analysis agent to use using multimodal LLM"""
        try:
            # Read and encode image to base64
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create multimodal decision chain
            decision_chain = self.medical_image_decision_prompt | self.decision_llm | self.medical_image_parser

            decision = decision_chain.invoke({
                "query": query,
                "image_base64": image_base64
            })

            normalized_agent = self._normalize_agent(decision.get("agent"))
            # Only allow medical image agents for this path
            if normalized_agent not in {"BRAIN_TUMOR_AGENT", "CHEST_XRAY_AGENT", "SKIN_LESION_AGENT"}:
                normalized_agent = "SKIN_LESION_AGENT"  # safe fallback

            normalized_conf = self._normalize_confidence(decision.get("confidence", 0.0))

            logger.info(f"Medical image agent decision: {normalized_agent} (confidence: {normalized_conf})")

            return {
                "status": "success",
                "agent": normalized_agent,
                "reasoning": decision["reasoning"],
                "confidence": normalized_conf,
                "image_type": decision["image_type"],
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in medical image agent decision: {str(e)}")
            return {
                "status": "error",
                "agent": "SKIN_LESION_AGENT",  # Fallback
                "reasoning": f"Error in medical image decision: {str(e)}",
                "confidence": 0.0,
                "image_type": "unknown",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def should_use_agent(self, decision: Dict[str, Any]) -> bool:
        """Check if the agent should be used based on confidence"""
        return decision["confidence"] >= config.AGENT_CONFIDENCE_THRESHOLD

# Global agent decision system instance
agent_decision_system = AgentDecisionSystem()