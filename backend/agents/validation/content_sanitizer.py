import logging
import bleach
import html
import re
from typing import Dict, Any, List, Optional

from config import config, logger

class ContentSanitizer:
    """Sanitizes content to remove harmful elements"""

    def __init__(self):
        """Initialize the content sanitizer"""
        # Allowed HTML tags for medical content
        self.allowed_tags = [
            'p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'ul', 'ol', 'li', 'blockquote', 'code', 'pre'
        ]

        # Allowed attributes
        self.allowed_attributes = {
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'title', 'width', 'height']
        }

        # Dangerous patterns to remove
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'<iframe[^>]*>.*?</iframe>',
            r'javascript:',
            r'vbscript:',
            r'onload\\s*=',
            r'onerror\\s*=',
            r'onclick\\s*='
        ]

    def sanitize_text(self, text: str) -> Dict[str, Any]:
        """Sanitize text content"""
        try:
            original_text = text

            # Remove dangerous patterns
            for pattern in self.dangerous_patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

            # HTML encode special characters
            text = html.escape(text)

            # Clean HTML tags
            text = bleach.clean(
                text,
                tags=self.allowed_tags,
                attributes=self.allowed_attributes,
                strip=True
            )

            # Remove excessive whitespace
            text = re.sub(r'\\s+', ' ', text).strip()

            return {
                'status': 'success',
                'original_text': original_text,
                'sanitized_text': text,
                'changes_made': original_text != text
            }

        except Exception as e:
            logger.error(f"Error in text sanitization: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'original_text': text,
                'sanitized_text': text,
                'changes_made': False
            }

    def sanitize_html(self, html_content: str) -> Dict[str, Any]:
        """Sanitize HTML content"""
        try:
            original_html = html_content

            # Clean HTML
            cleaned_html = bleach.clean(
                html_content,
                tags=self.allowed_tags,
                attributes=self.allowed_attributes,
                strip=True
            )

            return {
                'status': 'success',
                'original_html': original_html,
                'sanitized_html': cleaned_html,
                'changes_made': original_html != cleaned_html
            }

        except Exception as e:
            logger.error(f"Error in HTML sanitization: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'original_html': html_content,
                'sanitized_html': html_content,
                'changes_made': False
            }

    def remove_sensitive_data(self, text: str) -> Dict[str, Any]:
        """Remove sensitive data from text"""
        try:
            original_text = text

            # Remove email addresses
            text = re.sub(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', '[EMAIL]', text)

            # Remove phone numbers
            text = re.sub(r'\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b', '[PHONE]', text)

            # Remove SSN patterns
            text = re.sub(r'\\b\\d{3}-\\d{2}-\\d{4}\\b', '[SSN]', text)

            # Remove credit card patterns
            text = re.sub(r'\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b', '[CARD]', text)

            # Remove medical record IDs
            text = re.sub(r'\\b[A-Z]{2,3}\\d{6,8}\\b', '[MEDICAL_ID]', text)

            return {
                'status': 'success',
                'original_text': original_text,
                'sanitized_text': text,
                'changes_made': original_text != text
            }

        except Exception as e:
            logger.error(f"Error in sensitive data removal: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'original_text': text,
                'sanitized_text': text,
                'changes_made': False
            }
