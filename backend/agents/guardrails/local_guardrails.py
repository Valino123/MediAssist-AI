import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from config import config, logger

class LocalGuardrails:
    """Local guardrails for content safety and medical appropriateness"""

    def __init__(self):
        """Initialize the guardrails system"""
        # Medical emergency keywords
        self.emergency_keywords = [
            'emergency', 'urgent', 'critical', 'severe', 'life-threatening',
            'heart attack', 'stroke', 'bleeding', 'unconscious', "can't breathe",
            'chest pain', 'difficulty breathing', 'severe pain', 'overdose'
        ]

        # Inappropriate content keywords
        self.inappropriate_keywords = [
            'suicide', 'self-harm', 'kill myself', 'end my life',
            'harm others', 'violence', 'weapon', 'bomb', 'threat'
        ]

        # Medical disclaimer triggers
        self.disclaimer_triggers = [
            'diagnosis', 'treatment', 'medication', 'prescription',
            'surgery', 'therapy', 'cure', 'heal', 'fix'
        ]

        # Confidence thresholds
        self.emergency_threshold = 0.7
        self.inappropriate_threshold = 0.8
        self.disclaimer_threshold = 0.6

    def check_input(self, input_text: str) -> Tuple[bool, str]:
        """Check if input is safe and appropriate"""
        try:
            # Check for emergency situations
            emergency_result = self._check_emergency(input_text)
            if emergency_result[0]:
                return False, f"EMERGENCY DETECTED: {emergency_result[1]}"

            # Check for inappropriate content
            inappropriate_result = self._check_inappropriate_content(input_text)
            if inappropriate_result[0]:
                return False, f"INAPPROPRIATE CONTENT: {inappropriate_result[1]}"

            # Check for medical advice requests
            medical_result = self._check_medical_advice(input_text)
            if medical_result[0]:
                return True, f"MEDICAL DISCLAIMER REQUIRED: {medical_result[1]}"

            return True, "Input is safe and appropriate"

        except Exception as e:
            logger.error(f"Error in guardrails check: {str(e)}")
            return False, f"Guardrails error: {str(e)}"

    def check_output(self, output_text: str) -> Tuple[bool, str]:
        """Check if output is safe and appropriate"""
        try:
            # Check for inappropriate content in output
            inappropriate_result = self._check_inappropriate_content(output_text)
            if inappropriate_result[0]:
                return False, f"INAPPROPRIATE OUTPUT: {inappropriate_result[1]}"

            # Check for medical advice in output
            medical_result = self._check_medical_advice(output_text)
            if medical_result[0]:
                return True, f"MEDICAL DISCLAIMER REQUIRED: {medical_result[1]}"

            return True, "Output is safe and appropriate"

        except Exception as e:
            logger.error(f"Error in output guardrails check: {str(e)}")
            return False, f"Output guardrails error: {str(e)}"

    def _check_emergency(self, text: str) -> Tuple[bool, str]:
        """Check for emergency situations"""
        text_lower = text.lower()

        for keyword in self.emergency_keywords:
            if keyword in text_lower:
                return True, f"Emergency keyword detected: {keyword}"

        # Check for emergency patterns
        emergency_patterns = [
            r'i need help',
            r'call 911',
            r'emergency room',
            r'urgent care',
            r"can't breathe",
            r'chest hurts',
            r'severe pain'
        ]

        for pattern in emergency_patterns:
            if re.search(pattern, text_lower):
                return True, f"Emergency pattern detected: {pattern}"

        return False, ""

    def _check_inappropriate_content(self, text: str) -> Tuple[bool, str]:
        """Check for inappropriate content"""
        text_lower = text.lower()

        for keyword in self.inappropriate_keywords:
            if keyword in text_lower:
                return True, f"Inappropriate keyword detected: {keyword}"

        # Check for inappropriate patterns
        inappropriate_patterns = [
            r'i want to die',
            r'kill myself',
            r'harm myself',
            r'hurt others',
            r'violence',
            r'weapon',
            r'bomb',
            r'threat'
        ]

        for pattern in inappropriate_patterns:
            if re.search(pattern, text_lower):
                return True, f"Inappropriate pattern detected: {pattern}"

        return False, ""

    def _check_medical_advice(self, text: str) -> Tuple[bool, str]:
        """Check if medical advice is being requested or given"""
        text_lower = text.lower()

        for keyword in self.disclaimer_triggers:
            if keyword in text_lower:
                return True, f"Medical advice keyword detected: {keyword}"

        # Check for medical advice patterns
        medical_patterns = [
            r'should i take',
            r'what medication',
            r'how to treat',
            r"what's wrong with me",
            r'do i have',
            r'is this normal',
            r'when to see a doctor',
            r'emergency room',
            r'urgent care'
        ]

        for pattern in medical_patterns:
            if re.search(pattern, text_lower):
                return True, f"Medical advice pattern detected: {pattern}"

        return False, ""

    def get_emergency_response(self) -> str:
        """Get emergency response message"""
        return """🚨 EMERGENCY DETECTED 🚨

This appears to be a medical emergency. Please:

1. Call 911 immediately if you're in the United States
2. Go to your nearest emergency room
3. Contact your local emergency services

This AI assistant cannot provide emergency medical care. Please seek immediate professional medical attention.

For non-emergency medical questions, please consult with a healthcare professional."""

    def get_inappropriate_response(self) -> str:
        """Get inappropriate content response"""
        return """I cannot assist with this type of content.

If you're experiencing thoughts of self-harm or harm to others, please:
- Call the National Suicide Prevention Lifeline: 988
- Text HOME to 741741 for crisis support
- Contact your local emergency services: 911

For medical questions, please consult with a healthcare professional."""

    def get_medical_disclaimer(self) -> str:
        """Get medical disclaimer"""
        return """⚠️ MEDICAL DISCLAIMER ⚠️

This AI assistant is for informational purposes only and does not provide medical advice, diagnosis, or treatment.

Always consult with a qualified healthcare professional for:
- Medical diagnosis
- Treatment recommendations
- Medication advice
- Emergency situations

This information should not replace professional medical consultation."""
