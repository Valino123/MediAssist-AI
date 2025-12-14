"""
Medical policy guardrails
Enforces medical advice boundaries and emergency detection
"""
import re
from typing import Dict, Tuple

from config import logger


class MedicalGuardrails:
    """Enforces medical content policies"""

    # Emergency keywords
    EMERGENCY_KEYWORDS = [
        'emergency', 'urgent', 'critical', 'severe', 'life-threatening',
        'heart attack', 'stroke', 'bleeding', 'unconscious', "can't breathe",
        'chest pain', 'difficulty breathing', 'severe pain', 'overdose'
    ]

    # Crisis keywords
    CRISIS_KEYWORDS = [
        'suicide', 'self-harm', 'kill myself', 'end my life',
        'harm others', 'violence', 'weapon', 'bomb', 'threat'
    ]

    # Medical advice patterns
    DIAGNOSIS_PATTERN = re.compile(
        r'diagnose\s+me|what\'s\s+wrong\s+with\s+me|do\s+i\s+have',
        re.IGNORECASE
    )
    TREATMENT_PATTERN = re.compile(
        r'how\s+to\s+treat|should\s+i\s+take|what\s+medication',
        re.IGNORECASE
    )
    PRESCRIPTION_PATTERN = re.compile(
        r'prescription\s+for|prescribe\s+me',
        re.IGNORECASE
    )

    # Emergency patterns
    EMERGENCY_PATTERN = re.compile(
        r'call\s+911|emergency\s+room|urgent\s+care|can\'t\s+breathe|chest\s+hurts',
        re.IGNORECASE
    )
    CRISIS_PATTERN = re.compile(
        r'want\s+to\s+die|kill\s+myself|harm\s+myself',
        re.IGNORECASE
    )

    def check_input(self, text: str) -> Tuple[bool, str, str]:
        """
        Check input against medical policies
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (is_safe, category, reason)
        """
        try:
            # Check for emergencies (highest priority)
            is_emergency, reason = self._check_emergency(text)
            if is_emergency:
                return False, 'emergency', reason

            # Check for crisis situations
            is_crisis, reason = self._check_crisis(text)
            if is_crisis:
                return False, 'crisis', reason

            # Check for medical advice requests
            needs_disclaimer, reason = self._check_medical_advice(text)
            if needs_disclaimer:
                return True, 'disclaimer_required', reason

            return True, 'safe', ''

        except Exception as e:
            logger.error(f"Error in medical guardrails: {str(e)}")
            return False, 'error', str(e)

    def check_output(self, text: str) -> Tuple[bool, str]:
        """
        Check output for inappropriate medical content
        
        Args:
            text: Output text to check
            
        Returns:
            Tuple of (is_safe, reason)
        """
        try:
            # Check if output contains medical advice
            if self._has_medical_advice(text):
                return True, 'disclaimer_required'

            # Check for crisis content
            is_crisis, reason = self._check_crisis(text)
            if is_crisis:
                return False, reason

            return True, ''

        except Exception as e:
            logger.error(f"Error checking output: {str(e)}")
            return False, str(e)

    def _check_emergency(self, text: str) -> Tuple[bool, str]:
        """Check for emergency situations"""
        text_lower = text.lower()

        for keyword in self.EMERGENCY_KEYWORDS:
            if keyword in text_lower:
                return True, f"Emergency: {keyword}"

        if self.EMERGENCY_PATTERN.search(text):
            return True, "Emergency pattern detected"

        return False, ''

    def _check_crisis(self, text: str) -> Tuple[bool, str]:
        """Check for mental health crisis"""
        text_lower = text.lower()

        for keyword in self.CRISIS_KEYWORDS:
            if keyword in text_lower:
                return True, f"Crisis: {keyword}"

        if self.CRISIS_PATTERN.search(text):
            return True, "Crisis pattern detected"

        return False, ''

    def _check_medical_advice(self, text: str) -> Tuple[bool, str]:
        """Check if medical advice is requested"""
        if self.DIAGNOSIS_PATTERN.search(text):
            return True, "Diagnosis request"
        if self.TREATMENT_PATTERN.search(text):
            return True, "Treatment request"
        if self.PRESCRIPTION_PATTERN.search(text):
            return True, "Prescription request"

        return False, ''

    def _has_medical_advice(self, text: str) -> bool:
        """Check if text contains medical advice"""
        return (
            self.DIAGNOSIS_PATTERN.search(text) is not None or
            self.TREATMENT_PATTERN.search(text) is not None or
            self.PRESCRIPTION_PATTERN.search(text) is not None
        )

    def get_emergency_response(self) -> str:
        """Get emergency response message"""
        return """🚨 MEDICAL EMERGENCY 🚨

This appears to be a medical emergency. Please:
1. Call 911 immediately (United States)
2. Go to your nearest emergency room
3. Contact your local emergency services

This AI assistant cannot provide emergency medical care."""

    def get_crisis_response(self) -> str:
        """Get crisis response message"""
        return """I'm concerned about your safety. Please reach out for help:
- National Suicide Prevention Lifeline: 988
- Crisis Text Line: Text HOME to 741741
- Emergency Services: 911

You're not alone, and help is available."""

    def get_medical_disclaimer(self) -> str:
        """Get medical disclaimer"""
        return """⚠️ MEDICAL DISCLAIMER

This information is for educational purposes only and does not constitute medical advice.
Always consult with a qualified healthcare professional for medical concerns."""

