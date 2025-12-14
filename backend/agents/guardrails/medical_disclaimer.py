import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from config import config, logger

class MedicalDisclaimer:
    """Handles medical disclaimers and warnings"""

    def __init__(self):
        """Initialize the medical disclaimer system"""
        self.disclaimer_text = self._get_disclaimer_text()
        self.warning_text = self._get_warning_text()
        self.emergency_text = self._get_emergency_text()

    def get_disclaimer(self, context: str = "general") -> Dict[str, Any]:
        """Get appropriate disclaimer based on context"""
        try:
            if context == "emergency":
                return {
                    'status': 'success',
                    'disclaimer': self.emergency_text,
                    'type': 'emergency',
                    'timestamp': datetime.now().isoformat()
                }
            elif context == "medical_advice":
                return {
                    'status': 'success',
                    'disclaimer': self.warning_text,
                    'type': 'medical_advice',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'success',
                    'disclaimer': self.disclaimer_text,
                    'type': 'general',
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Error getting disclaimer: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'disclaimer': self.disclaimer_text,
                'type': 'general',
                'timestamp': datetime.now().isoformat()
            }

    def _get_disclaimer_text(self) -> str:
        """Get general medical disclaimer"""
        return """⚠️ MEDICAL DISCLAIMER ⚠️

This AI assistant is for informational purposes only and does not provide medical advice, diagnosis, or treatment.

Important Notes:
• Always consult with a qualified healthcare professional
• This information should not replace professional medical consultation
• For medical emergencies, call 911 or go to your nearest emergency room
• Individual medical conditions may vary

By using this service, you acknowledge that you understand these limitations."""

    def _get_warning_text(self) -> str:
        """Get medical advice warning"""
        return """🚨 MEDICAL ADVICE WARNING 🚨

This AI assistant cannot provide medical advice, diagnosis, or treatment recommendations.

For medical concerns:
• Consult with a qualified healthcare professional
• Schedule an appointment with your doctor
• Contact your local healthcare provider
• For emergencies, call 911 immediately

This information is for educational purposes only and should not be used for medical decision-making."""

    def _get_emergency_text(self) -> str:
        """Get emergency response text"""
        return """🚨 MEDICAL EMERGENCY 🚨

This appears to be a medical emergency. Please:

1. Call 911 immediately (United States)
2. Go to your nearest emergency room
3. Contact your local emergency services

This AI assistant cannot provide emergency medical care.

For non-emergency medical questions, please consult with a healthcare professional."""

    def add_disclaimer_to_response(self, response: str, context: str = "general") -> str:
        """Add appropriate disclaimer to response"""
        try:
            disclaimer = self.get_disclaimer(context)

            if disclaimer['status'] == 'success':
                return f"{response}\\n\\n---\\n\\n{disclaimer['disclaimer']}"
            else:
                return f"{response}\\n\\n---\\n\\n{self.disclaimer_text}"

        except Exception as e:
            logger.error(f"Error adding disclaimer: {str(e)}")
            return f"{response}\\n\\n---\\n\\n{self.disclaimer_text}"
