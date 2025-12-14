"""
PII (Personal Identifiable Information) validation module
Detects and sanitizes sensitive personal data
"""
import re
from typing import Dict, Any, List

from config import logger


class PIIValidator:
    """Detects and sanitizes personally identifiable information"""
    
    # PII patterns
    EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    SSN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    CREDIT_CARD = re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b')
    MEDICAL_ID = re.compile(r'\b[A-Z]{2,3}\d{6,8}\b')

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect PII in text
        
        Args:
            text: Text to check
            
        Returns:
            Dictionary with detected PII types
        """
        try:
            detected = []
            
            if self.EMAIL.search(text):
                detected.append('email')
            if self.PHONE.search(text):
                detected.append('phone')
            if self.SSN.search(text):
                detected.append('ssn')
            if self.CREDIT_CARD.search(text):
                detected.append('credit_card')
            if self.MEDICAL_ID.search(text):
                detected.append('medical_id')
            
            return {
                'has_pii': len(detected) > 0,
                'types': detected
            }
        
        except Exception as e:
            logger.error(f"Error detecting PII: {str(e)}")
            return {'has_pii': False, 'types': []}

    def sanitize(self, text: str) -> str:
        """
        Remove PII from text
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text with PII replaced by placeholders
        """
        try:
            text = self.EMAIL.sub('[EMAIL]', text)
            text = self.PHONE.sub('[PHONE]', text)
            text = self.SSN.sub('[SSN]', text)
            text = self.CREDIT_CARD.sub('[CARD]', text)
            text = self.MEDICAL_ID.sub('[MEDICAL_ID]', text)
            return text
        
        except Exception as e:
            logger.error(f"Error sanitizing PII: {str(e)}")
            return text

