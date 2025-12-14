"""
Spam detection module
Identifies spam and abusive content patterns
"""
import re
from typing import Dict, Any

from config import logger


class SpamValidator:
    """Detects spam and abusive content"""
    
    # Spam patterns
    MARKETING = re.compile(
        r'click\s+here|buy\s+now|limited\s+time|act\s+now|free\s+money|win\s+prize',
        re.IGNORECASE
    )
    REPETITION = re.compile(r'(.)\1{5,}')
    EXCESSIVE_PUNCT = re.compile(r'[!?]{3,}')
    
    # Thresholds
    CAPS_THRESHOLD = 0.7
    WORD_REPEAT_THRESHOLD = 0.3

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect spam patterns
        
        Args:
            text: Text to check
            
        Returns:
            Dictionary with spam detection results
        """
        try:
            issues = []
            
            if self.MARKETING.search(text):
                issues.append("Marketing spam detected")
            
            if self.REPETITION.search(text):
                issues.append("Excessive repetition detected")
            
            if self.EXCESSIVE_PUNCT.search(text):
                issues.append("Excessive punctuation detected")
            
            if self._has_excessive_caps(text):
                issues.append("Excessive capitalization detected")
            
            if self._has_repeated_words(text):
                issues.append("Word repetition detected")
            
            return {
                'is_spam': len(issues) > 0,
                'issues': issues
            }
        
        except Exception as e:
            logger.error(f"Error detecting spam: {str(e)}")
            return {'is_spam': False, 'issues': []}

    def _has_excessive_caps(self, text: str) -> bool:
        """Check for excessive capitalization"""
        if len(text) < 10:
            return False
        caps_count = sum(1 for c in text if c.isupper())
        return (caps_count / len(text)) > self.CAPS_THRESHOLD

    def _has_repeated_words(self, text: str) -> bool:
        """Check for excessive word repetition"""
        words = text.lower().split()
        if len(words) < 10:
            return False
        
        word_counts = {}
        for word in words:
            if len(word) > 3:
                word_counts[word] = word_counts.get(word, 0) + 1
                if word_counts[word] > len(words) * self.WORD_REPEAT_THRESHOLD:
                    return True
        return False

