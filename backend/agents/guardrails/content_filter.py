import logging
import re
import warnings
from typing import Dict, Any, List, Optional

# Suppress pkg_resources deprecation warning from profanity_check
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

try:
    from profanity_check import predict, predict_prob  # external optional
    _PROFANITY_AVAILABLE = True
except Exception:
    _PROFANITY_AVAILABLE = False

from config import config, logger

class ContentFilter:
    """Filters inappropriate and harmful content"""

    def __init__(self):
        """Initialize the content filter"""
        self.profanity_enabled = config.ENABLE_PROFANITY_FILTER

        # Custom medical content filters
        self.medical_filters = [
            r'prescription\\s+for\\s+[a-zA-Z]+',
            r'diagnose\\s+me\\s+with',
            r'treat\\s+my\\s+[a-zA-Z]+',
            r'cure\\s+my\\s+[a-zA-Z]+',
            r'heal\\s+my\\s+[a-zA-Z]+'
        ]

        # Spam detection patterns
        self.spam_patterns = [
            r'click\\s+here',
            r'buy\\s+now',
            r'limited\\s+time',
            r'act\\s+now',
            r'free\\s+money',
            r'win\\s+prize',
            r'congratulations\\s+you\\s+won'
        ]

    def filter_content(self, text: str) -> Dict[str, Any]:
        """Filter content for appropriateness"""
        try:
            filtered_text = text
            issues = []
            warnings = []

            # Check profanity
            if self.profanity_enabled:
                profanity_result = self._check_profanity(text)
                if profanity_result['has_profanity']:
                    issues.append("Profanity detected")
                    filtered_text = self._replace_profanity(filtered_text)

            # Check medical content
            medical_result = self._check_medical_content(text)
            if medical_result['has_medical_advice']:
                warnings.append("Medical advice content detected")

            # Check spam
            spam_result = self._check_spam(text)
            if spam_result['is_spam']:
                issues.append("Spam content detected")

            # Check for personal information
            personal_result = self._check_personal_info(text)
            if personal_result['has_personal_info']:
                warnings.append("Personal information detected")
                filtered_text = self._remove_personal_info(filtered_text)

            return {
                'status': 'success',
                'original_text': text,
                'filtered_text': filtered_text,
                'issues': issues,
                'warnings': warnings,
                'is_clean': len(issues) == 0,
                'requires_review': len(warnings) > 0
            }

        except Exception as e:
            logger.error(f"Error in content filtering: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'original_text': text,
                'filtered_text': text,
                'issues': ['Filtering error'],
                'warnings': [],
                'is_clean': False,
                'requires_review': True
            }

    def _check_profanity(self, text: str) -> Dict[str, Any]:
        """Check for profanity using profanity-check library"""
        try:
            if not self.profanity_enabled or not _PROFANITY_AVAILABLE:
                return {'has_profanity': False, 'confidence': 0.0}

            # Check for profanity
            has_profanity = predict([text])[0] == 1
            confidence = predict_prob([text])[0]

            return {
                'has_profanity': has_profanity,
                'confidence': confidence
            }

        except Exception as e:
            logger.error(f"Error checking profanity: {str(e)}")
            return {'has_profanity': False, 'confidence': 0.0}

    def _replace_profanity(self, text: str) -> str:
        """Replace profanity with asterisks"""
        # Simple profanity replacement (in production, use a more sophisticated approach)
        profanity_words = ['damn', 'hell', 'crap', 'stupid', 'idiot']

        for word in profanity_words:
            pattern = re.compile(r'\\b' + re.escape(word) + r'\\b', re.IGNORECASE)
            text = pattern.sub('*' * len(word), text)

        return text

    def _check_medical_content(self, text: str) -> Dict[str, Any]:
        """Check for medical advice content"""
        text_lower = text.lower()

        for pattern in self.medical_filters:
            if re.search(pattern, text_lower):
                return {'has_medical_advice': True, 'pattern': pattern}

        return {'has_medical_advice': False, 'pattern': None}

    def _check_spam(self, text: str) -> Dict[str, Any]:
        """Check for spam content"""
        text_lower = text.lower()

        for pattern in self.spam_patterns:
            if re.search(pattern, text_lower):
                return {'is_spam': True, 'pattern': pattern}

        # Check for excessive repetition
        words = text_lower.split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                if word_counts[word] > len(words) * 0.3:
                    return {'is_spam': True, 'pattern': 'excessive_repetition'}

        return {'is_spam': False, 'pattern': None}

    def _check_personal_info(self, text: str) -> Dict[str, Any]:
        """Check for personal information"""
        # Email pattern
        email_pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
        # Phone pattern
        phone_pattern = r'\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b'
        # SSN pattern
        ssn_pattern = r'\\b\\d{3}-\\d{2}-\\d{4}\\b'

        has_email = bool(re.search(email_pattern, text))
        has_phone = bool(re.search(phone_pattern, text))
        has_ssn = bool(re.search(ssn_pattern, text))

        return {
            'has_personal_info': has_email or has_phone or has_ssn,
            'has_email': has_email,
            'has_phone': has_phone,
            'has_ssn': has_ssn
        }

    def _remove_personal_info(self, text: str) -> str:
        """Remove personal information from text"""
        # Remove email addresses
        text = re.sub(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', '[EMAIL]', text)

        # Remove phone numbers
        text = re.sub(r'\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b', '[PHONE]', text)

        # Remove SSN
        text = re.sub(r'\\b\\d{3}-\\d{2}-\\d{4}\\b', '[SSN]', text)

        return text
