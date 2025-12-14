import os
import logging
from typing import List, Dict, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI 
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from config import config, logger

class SimpleChatAgent:
    
    def __init__(self):
        """Initialize the chat agent"""
        self.conversation_history : List[Dict] = []
        self.system_prompt = config.CHAT_SYSTEM_PROMPT
        self.llm = None
        self.llm_available = False
        
        # Try to initialize LLM if API key is available
        if config.DASHSCOPE_API_KEY :
            try:
                self.llm = ChatOpenAI(
                    model=config.QWEN_MODEL,
                    temperature=float(config.TEMPERATURE or "0.7"),
                    max_tokens=int(config.QWEN_MAX_TOKENS or "1000"),
                    openai_api_key=config.DASHSCOPE_API_KEY,
                    base_url=config.BASE_URL,
                )
                self.llm_available = True
                logger.info(f"SimpleChatAgent initialized with LLM model: {config.QWEN_MODEL}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {str(e)}. Using fallback mode.")
                self.llm_available = False
        else:
            logger.info("No API key provided. Using fallback mode.")
            self.llm_available = False

    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.conversation_history.append(message)

        if len(self.conversation_history) > config.MAX_CONVERSATION_HISTORY * 2:
            self.conversation_history=self.conversation_history[-config.MAX_CONVERSATION_HISTORY*2:]

    def get_conversation_context(self) -> str:
        """Get formatted conversation context"""
        if not self.conversation_history:
            return ""
        
        context = "Previous conversation:\n"
        for msg in self.conversation_history[-6:]:
            context += f"{msg['role'].title()}: {msg['content']}\n"
        return context
    
    def process_message(self, user_message: str) -> Dict:
        """Process a user message and return AI response"""
        try:
            # Add user message to history
            self.add_message("user", user_message)

            # Check if LLM is available
            if not self.llm_available or not self.llm:
                # Fallback response when LLM is not available
                ai_response = self._get_fallback_response(user_message)
            else:
                # Use LLM for response
                ai_response = self._get_llm_response(user_message)
            
            # Add AI response to history
            self.add_message("assistant", ai_response)
            
            logger.info(f"Chat response generated for user message: {user_message[:50]}...")
            
            return {
                "status": "success",
                "response": ai_response,
                "agent": "SimpleChatAgent" if self.llm_available else "FallbackMode",
                "timestamp": datetime.now().isoformat(),
                "conversation_length": len(self.conversation_history)
            }
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "status": "error",
                "response": "I'm sorry, I encountered an error processing your message. Please try again.",
                "agent": "ERROR",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _get_llm_response(self, user_message: str) -> str:
        """Get response from LLM"""
        try:
            # Get conversation context
            conversation_context = self.get_conversation_context()
            
            prompt = f"""System: {self.system_prompt}

{conversation_context}

User: {user_message}

Assistant:"""
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            
            # Extract response content
            if hasattr(response, "content"):
                return response.content
            else:
                return str(response)
        except Exception as e:
            logger.error(f"LLM error: {str(e)}")
            return self._get_fallback_response(user_message)
    
    def _get_fallback_response(self, user_message: str) -> str:
        """Get fallback response when LLM is not available"""
        user_lower = user_message.lower()
        
        # Simple keyword-based responses
        if any(word in user_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm your medical assistant. I can help you with health-related questions. Note: I'm currently running in fallback mode. To enable AI responses, please configure your API key in the .env file."
        
        elif any(word in user_lower for word in ["symptom", "pain", "hurt", "ache"]):
            return "I understand you're experiencing symptoms. While I can provide general health information, please consult with a healthcare professional for proper medical advice and diagnosis."
        
        elif any(word in user_lower for word in ["medicine", "medication", "drug", "pill"]):
            return "For medication-related questions, it's important to consult with a healthcare provider or pharmacist. They can provide personalized advice based on your specific situation."
        
        elif any(word in user_lower for word in ["emergency", "urgent", "help"]):
            return "If this is a medical emergency, please call emergency services immediately (911 in the US, 999 in the UK, etc.). For urgent medical concerns, contact your healthcare provider or visit the nearest emergency room."
        
        elif any(word in user_lower for word in ["thank", "thanks"]):
            return "You're welcome! I'm here to help with any health-related questions you might have."
        
        else:
            return f"I received your message: '{user_message}'. I'm currently running in fallback mode. To enable AI-powered responses, please configure your API key in the .env file. For now, I can provide general health information and guidance."
        
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict]:
        """Get the conversation history"""
        return self.conversation_history

    def get_history_summary(self) -> Dict:
        """Get summary of conversation history"""
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": len([msg for msg in self.conversation_history if msg["role"] == "user"]),
            "assistant_messages": len([msg for msg in self.conversation_history if msg["role"] == "assistant"]),
            "last_message_time": self.conversation_history[-1]["timestamp"] if self.conversation_history else None
        }

# Global chat agent instance
chat_agent = SimpleChatAgent()