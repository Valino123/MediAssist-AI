import logging
from typing import Dict, List, Optional, Union, Any
import base64
from datetime import datetime

from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver

from config import config, logger
from .agent_decision import AgentDecisionSystem, AgentState 
from .simple_chat_agent import chat_agent
from .rag_agent import rag_agent 
# Use global medical agents defined in their modules
from .image_analysis_agent.brain_tumor_agent import BrainTumorAgent as _BTClass  # type: ignore
from .image_analysis_agent.chest_xray_agent import ChestXrayAgent as _CXClass  # type: ignore
from .image_analysis_agent.skin_lesion_agent import SkinLesionAgent as _SLClass  # type: ignore
from .web_search_agent import WebSearchAgent
from .guardrails import LocalGuardrails, ContentFilter
from .validation import InputValidator

class MultiAgentOrchestrator: 

    def __init__(self):
        """Initialize the multi-agent orchestrator"""
        self.agent_decision_system = AgentDecisionSystem()
        # self.memory = MemorySaver()
        
        # Initialize lightweight/shared helpers
        self.guardrails = LocalGuardrails()
        self.content_filter = ContentFilter()
        self.validator = InputValidator()
        self.web_search = WebSearchAgent()

        # Initialize medical image analysis agents (singletons per process)
        self.brain_tumor_agent = _BTClass()
        self.chest_xray_agent = _CXClass()
        self.skin_lesion_agent = _SLClass()
       
        # Create the workflow graph
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph: 
        """Create the LangGraph workflow with medical image routing"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_input", self._analyze_input)
        workflow.add_node("input_guardrails", self._input_guardrails)
        workflow.add_node("decide_agent", self._decide_agent)
        workflow.add_node("conversation_agent", self._run_conversation_agent)
        workflow.add_node("rag_agent", self._run_rag_agent)
        workflow.add_node("web_search_agent", self._run_web_search_agent)
        workflow.add_node("image_analysis_agent", self._run_image_analysis_agent)
        workflow.add_node("brain_tumor_agent", self._run_brain_tumor_agent)
        workflow.add_node("chest_xray_agent", self._run_chest_xray_agent)
        workflow.add_node("skin_lesion_agent", self._run_skin_lesion_agent)
        # New orchestration nodes per architecture
        workflow.add_node("confidence_check", self._confidence_check)
        workflow.add_node("human_validation", self._human_validation)
        workflow.add_node("output_guardrails", self._apply_output_guardrails)

        # Set entry point
        workflow.set_entry_point("analyze_input")

        # Add edges
        workflow.add_edge("analyze_input", "input_guardrails")
        workflow.add_edge("input_guardrails", "decide_agent")

        # Add conditional edges based on agent decision
        workflow.add_conditional_edges(
            "decide_agent",
            self._route_to_agent,
            {
                "CONVERSATION_AGENT": "conversation_agent",
                "RAG_AGENT": "rag_agent",
                "WEB_SEARCH_AGENT": "web_search_agent",
                "IMAGE_ANALYSIS_AGENT": "image_analysis_agent",
            }
        )

        # Add conditional edges from image_analysis_agent to specific medical agents
        workflow.add_conditional_edges(
            "image_analysis_agent",
            self._route_medical_image_agent,
            {
                "BRAIN_TUMOR_AGENT": "brain_tumor_agent",
                "CHEST_XRAY_AGENT": "chest_xray_agent",
                "SKIN_LESION_AGENT": "skin_lesion_agent",
            }
        )
        
        # LLM + RAG branch with confidence routing and guardrails
        workflow.add_edge("rag_agent", "confidence_check")
        workflow.add_conditional_edges(
            "confidence_check",
            self._route_confidence,
            {
                "HIGH": "output_guardrails",
                "LOW": "web_search_agent",
            },
        )

        # Web search flows to guardrails
        workflow.add_edge("web_search_agent", "output_guardrails")

        # Conversation flows directly to guardrails
        workflow.add_edge("conversation_agent", "output_guardrails")

        # Medical image path → specific agent → human validation → guardrails
        workflow.add_edge("brain_tumor_agent", "human_validation")
        workflow.add_edge("chest_xray_agent", "human_validation")
        workflow.add_edge("skin_lesion_agent", "human_validation")
        workflow.add_edge("human_validation", "output_guardrails")

        # Final output
        workflow.add_edge("output_guardrails", END)

        # return workflow.compile(checkpointer=self.memory)
        return workflow.compile()


    def _analyze_input(self, state: AgentState) -> AgentState:
        """Analyze the input to determine if it contains images"""
        # For now, we'll assume no images (Part 6 will add image analysis)
        # This will be enhanced to detect uploaded images
        state.has_image = False 
        state.image_type = None 
        state.image_path = None
        return state 

    def _input_guardrails(self, state: AgentState) -> AgentState:
        """Run validation and input guardrails. May short-circuit for emergencies/inappropriate."""
        try:
            text = state.current_input or ""
            # Basic validation
            v = self.validator.validate_text_input(text)
            if not v.is_valid:
                state.agent_response = "Input validation failed: " + "; ".join(v.errors)
                state.selected_agent = "CONVERSATION_AGENT"
                return state

            # Guardrails check
            ok, msg = self.guardrails.check_input(text)
            if not ok:
                # Emergency or inappropriate: return immediate response
                if msg.startswith("EMERGENCY"):
                    state.agent_response = self.guardrails.get_emergency_response()
                elif msg.startswith("INAPPROPRIATE"):
                    state.agent_response = self.guardrails.get_inappropriate_response()
                else:
                    state.agent_response = msg
                state.selected_agent = "CONVERSATION_AGENT"
                # Route directly to output guardrails by setting a terminal response
                # The workflow graph still moves to decide_agent; we keep response for final stage
            return state
        except Exception:
            return state

    def _decide_agent(self, state: AgentState) -> AgentState:
        """Decide which agent should handle the query"""
        decision = self.agent_decision_system.decide_agent(
            query=state.current_input,
            has_image=state.has_image,
            image_type=state.image_type
        )

        state.selected_agent = decision["agent"]
        state.confidence = decision["confidence"]
        return state

    def _route_to_agent(self, state: AgentState) -> str:
        """Route to the appropriate agent based on decision"""
        return state.selected_agent

    def _run_conversation_agent(self, state: AgentState) -> AgentState:
        """Run the conversation agent"""
        try:
            result = chat_agent.process_message(state.current_input)
            state.agent_response = result["response"]
            state.selected_agent = "CONVERSATION_AGENT"
        except Exception as e:
            logger.error(f"Error in conversation agent: {str(e)}")
            state.agent_response = "I'm sorry, I encountered an error. Please try again."
            state.selected_agent = "CONVERSATION_AGENT"

        return state

    def _run_rag_agent(self, state: AgentState) -> AgentState:
        """Run the RAG agent"""
        try:
            # Call RAG using its internal/global vector store
            result = rag_agent.process_query(state.current_input)
            state.agent_response = result["response"]
            state.selected_agent = "RAG_AGENT"
            # Use RAG agent's confidence for routing decisions
            state.rag_confidence = result.get("confidence", 0.0)
            
            # Log confidence for debugging
            logger.info(f"RAG agent confidence: {state.rag_confidence}, threshold: {config.RAG_CONFIDENCE_THRESHOLD}")
            
        except Exception as e:
            logger.error(f"Error in RAG agent: {str(e)}")
            state.agent_response = "I'm sorry, I encountered an error retrieving information. Please try again."
            state.selected_agent = "RAG_AGENT"
            state.rag_confidence = 0.0

        return state

    def _confidence_check(self, state: AgentState) -> AgentState:
        """Check confidence of upstream agent (e.g., RAG) for routing"""
        try:
            # Ensure confidence is a float; default to 0.0 if missing
            state.confidence = float(state.confidence or 0.0)
        except Exception:
            state.confidence = 0.0
        return state

    def _route_confidence(self, state: AgentState) -> str:
        """Route based on confidence threshold"""
        threshold = float(getattr(config, "RAG_CONFIDENCE_THRESHOLD", 0.0) or 0.0)
        # Use RAG confidence for routing, fallback to general confidence
        confidence = float(getattr(state, 'rag_confidence', state.confidence) or 0.0)
        
        # Log routing decision for debugging
        logger.info(f"Confidence routing: confidence={confidence}, threshold={threshold}, routing={'HIGH' if confidence >= threshold else 'LOW'}")
        
        return "HIGH" if confidence >= threshold else "LOW"

    def _run_web_search_agent(self, state: AgentState) -> AgentState:
        """Run the web search agent using Google CSE and/or PubMed."""
        try:
            query = state.current_input or ""
            # Respect upstream preference if provided; fallback to heuristic
            search_type = (state.search_type or "").lower().strip() or ("literature" if any(k in query.lower() for k in ["study", "pubmed", "paper", "clinical"]) else "general")
            
            # Log web search activation
            logger.info(f"Web search agent activated for query: '{query}', search_type: {search_type}")
            
            result = self.web_search.search_medical_info(query, search_type)
            if result.get("status") == "success":
                # Get the formatted results from the web search agent
                formatted_results = result.get("results", {})
                
                # Handle different result structures based on search type
                if search_type == "literature":
                    # PubMed results structure
                    articles = formatted_results.get("articles", [])
                    summary = formatted_results.get("summary", "")
                    lines = [f"Medical literature search for '{query}':", summary]
                    
                    for i, article in enumerate(articles[:5], 1):
                        title = article.get("title", "")
                        authors = article.get("authors", "")
                        journal = article.get("journal", "")
                        year = article.get("year", "")
                        abstract = article.get("abstract", "")
                        url = article.get("url", "")
                        lines.append(f"{i}. {title}")
                        if authors:
                            lines.append(f"   Authors: {authors}")
                        if journal and year:
                            lines.append(f"   Journal: {journal} ({year})")
                        if abstract:
                            lines.append(f"   Abstract: {abstract}")
                        if url:
                            lines.append(f"   Link: {url}")
                        lines.append("")
                else:
                    # General web search results structure
                    sources = formatted_results.get("sources", [])
                    answer = formatted_results.get("answer", "")
                    summary = formatted_results.get("summary", "")
                    
                    lines = [f"Web search results for '{query}':", summary]
                    
                    if answer:
                        lines.append(f"Answer: {answer}")
                        lines.append("")
                    
                    for i, source in enumerate(sources[:5], 1):
                        title = source.get("title", "")
                        url = source.get("url", "")
                        content = source.get("content", "")
                        domain = source.get("domain", "")
                        lines.append(f"{i}. {title}")
                        if domain:
                            lines.append(f"   Source: {domain}")
                        if content:
                            lines.append(f"   Content: {content}")
                        if url:
                            lines.append(f"   Link: {url}")
                        lines.append("")
                
                state.agent_response = "\n".join(lines) if lines else "No relevant results found."
                logger.info(f"Web search completed successfully for query: '{query}'")
            else:
                state.agent_response = f"Web search error: {result.get('error', 'Unknown error')}"
                logger.error(f"Web search failed for query: '{query}', error: {result.get('error', 'Unknown error')}")
            state.selected_agent = "WEB_SEARCH_AGENT"
        except Exception as e:
            logger.error(f"Error in web search agent: {str(e)}")
            state.agent_response = "I'm sorry, I encountered an error with web search. Please try again."
            state.selected_agent = "WEB_SEARCH_AGENT"
        return state

    def _run_image_analysis_agent(self, state: AgentState) -> AgentState:
        """Run the image analysis agent - decides which specific medical agent to use"""
        try:
            if state.has_image and state.image_path:
                # Use multimodal LLM to decide which specific medical agent to use
                decision = self.agent_decision_system.decide_medical_image_agent(
                    state.current_input, 
                    state.image_path
                )
                
                # Store the decision for routing
                state.selected_agent = decision["agent"]
                state.confidence = decision["confidence"]
                state.image_type = decision.get("image_type", "unknown")
                
                logger.info(f"Medical image routed to: {decision['agent']} (confidence: {decision['confidence']})")
            else:
                state.agent_response = "Please upload a medical image for analysis."
                state.selected_agent = "IMAGE_ANALYSIS_AGENT"
        except Exception as e:
            logger.error(f"Error in image analysis agent: {str(e)}")
            state.agent_response = "I'm sorry, I encountered an error analyzing the image. Please try again."
            state.selected_agent = "IMAGE_ANALYSIS_AGENT"

        return state

    def _route_medical_image_agent(self, state: AgentState) -> str:
        """Route to the specific medical image analysis agent"""
        return state.selected_agent

    def _run_brain_tumor_agent(self, state: AgentState) -> AgentState:
        """Run the brain tumor analysis agent"""
        try:
            if state.has_image and state.image_path:
                # Read image file and build base64 data URI expected by agents
                with open(state.image_path, "rb") as f:
                    data_uri = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
                result = self.brain_tumor_agent.process_image(data_uri)
                # Prefer analysis_report if present, else stringified result
                state.agent_response = result.get("analysis_report") or result.get("response") or str(result)
                state.selected_agent = "BRAIN_TUMOR_AGENT"
            else:
                state.agent_response = "Please upload a brain MRI image for analysis."
                state.selected_agent = "BRAIN_TUMOR_AGENT"
        except Exception as e:
            logger.error(f"Error in brain tumor agent: {str(e)}")
            state.agent_response = "I'm sorry, I encountered an error analyzing the brain MRI image. Please try again."
            state.selected_agent = "BRAIN_TUMOR_AGENT"

        return state

    def _run_chest_xray_agent(self, state: AgentState) -> AgentState:
        """Run the chest X-ray analysis agent"""
        try:
            if state.has_image and state.image_path:
                with open(state.image_path, "rb") as f:
                    data_uri = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
                result = self.chest_xray_agent.process_image(data_uri)
                state.agent_response = result.get("analysis_report") or result.get("response") or str(result)
                state.selected_agent = "CHEST_XRAY_AGENT"
            else:
                state.agent_response = "Please upload a chest X-ray image for analysis."
                state.selected_agent = "CHEST_XRAY_AGENT"
        except Exception as e:
            logger.error(f"Error in chest X-ray agent: {str(e)}")
            state.agent_response = "I'm sorry, I encountered an error analyzing the chest X-ray image. Please try again."
            state.selected_agent = "CHEST_XRAY_AGENT"

        return state

    def _run_skin_lesion_agent(self, state: AgentState) -> AgentState:
        """Run the skin lesion analysis agent"""
        try:
            if state.has_image and state.image_path:
                with open(state.image_path, "rb") as f:
                    data_uri = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
                result = self.skin_lesion_agent.process_image(data_uri)
                state.agent_response = result.get("analysis_report") or result.get("response") or str(result)
                state.selected_agent = "SKIN_LESION_AGENT"
            else:
                state.agent_response = "Please upload a skin lesion image for analysis."
                state.selected_agent = "SKIN_LESION_AGENT"
        except Exception as e:
            logger.error(f"Error in skin lesion agent: {str(e)}")
            state.agent_response = "I'm sorry, I encountered an error analyzing the skin lesion image. Please try again."
            state.selected_agent = "SKIN_LESION_AGENT"

        return state

    def _human_validation(self, state: AgentState) -> AgentState:
        """Optional human-in-the-loop validation step (no-op placeholder).
        Integrations can hook a review UI here; orchestration simply passes through."""
        return state

    def _apply_output_guardrails(self, state: AgentState) -> AgentState:
        """Apply content filtering and append medical disclaimer before returning"""
        try:
            response_text = (state.agent_response or "").strip()

            # Content filter pass
            cf = self.content_filter.filter_content(response_text)
            if cf.get("status") == "success":
                response_text = cf.get("filtered_text") or response_text

            # Always append disclaimer once
            disclaimer = (
                "\n\nNote: This information is for educational purposes and is not a substitute "
                "for professional medical advice. For urgent or personal medical concerns, "
                "consult a qualified clinician."
            )
            if disclaimer not in response_text:
                response_text = f"{response_text}{disclaimer}"

            state.agent_response = response_text
        except Exception:
            # If guardrails fail, keep original response
            pass
        return state

    
    def process_query(self, query: str, vector_store, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Process a query through the multi-agent system"""
        try:
            # Store vector_store for use in agent methods
            self.vector_store = vector_store
            
            # Extract optional search type from any system meta message
            search_type = None
            for m in (conversation_history or []):
                try:
                    if m.get("role") == "system" and isinstance(m.get("content"), str) and m["content"].startswith("SEARCH_TYPE="):
                        search_type = m["content"].split("=", 1)[1].strip()
                        break
                except Exception:
                    pass

            # Create initial state
            state = AgentState(
                current_input=query,
                messages=conversation_history or [],
                search_type=search_type,
                timestamp=datetime.now().isoformat()
            )

            # Run the workflow
            result = self.workflow.invoke(state)

            return {
                "status": "success",
                "response": result.get("agent_response", "No response generated"),
                "agent": result.get("selected_agent", "UNKNOWN"),
                "confidence": result.get("confidence", 0.0),
                "timestamp": result.get("timestamp", datetime.now().isoformat())
            }
        except Exception as e:
            logger.error(f"Error in multi-agent orchestration: {str(e)}")
            return {
                "status": "error",
                "response": "I'm sorry, I encountered an error processing your request. Please try again.",
                "agent": "ERROR",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }

    def process_image_query(self, query: str, image_path: str, vector_store) -> Dict[str, Any]:
        """Process an image query through the multi-agent system with LangGraph workflow"""
        try:
            # Store vector_store for use in agent methods
            self.vector_store = vector_store
            
            # Create initial state with image information
            state = AgentState(
                current_input=query,
                has_image=True,
                image_path=image_path,
                timestamp=datetime.now().isoformat()
            )
            
            # Run the workflow - it will automatically route through IMAGE_ANALYSIS_AGENT
            # and then to the appropriate specific medical agent
            result = self.workflow.invoke(state)
            
            return {
                "status": "success",
                "response": result.get("agent_response", "No response generated"),
                "agent": result.get("selected_agent", "UNKNOWN"),
                "confidence": result.get("confidence", 0.0),
                "image_type": result.get("image_type", "unknown"),
                "timestamp": result.get("timestamp", datetime.now().isoformat())
            }
            
        except Exception as e:
            logger.error(f"Error in image query processing: {str(e)}")
            return {
                "status": "error",
                "response": "I'm sorry, I encountered an error analyzing the image. Please try again.",
                "agent": "ERROR",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }

# Global multi-agent orchestrator instance
multi_agent_orchestrator = MultiAgentOrchestrator()