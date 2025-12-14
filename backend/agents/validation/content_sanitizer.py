"""
HTML/Text content sanitization module
Removes harmful HTML elements and scripts
"""
import re
import bleach
import html
from typing import Dict, Any

from config import logger


class ContentSanitizer:
    """Sanitizes HTML and text content"""

    # XSS patterns for text sanitization
    XSS_PATTERNS = [
        (re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL), ''),
        (re.compile(r'<iframe[^>]*>.*?</iframe>', re.IGNORECASE | re.DOTALL), ''),
        (re.compile(r'javascript:', re.IGNORECASE), ''),
        (re.compile(r'vbscript:', re.IGNORECASE), ''),
        (re.compile(r'onload\s*=', re.IGNORECASE), ''),
        (re.compile(r'onerror\s*=', re.IGNORECASE), ''),
        (re.compile(r'onclick\s*=', re.IGNORECASE), ''),
    ]

    # Allowed HTML tags
    ALLOWED_TAGS = [
        'p', 'br', 'strong', 'em', 'u',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'ul', 'ol', 'li', 'blockquote', 'code', 'pre'
    ]

    # Allowed attributes
    ALLOWED_ATTRS = {
        'a': ['href', 'title'],
        'img': ['src', 'alt', 'title', 'width', 'height']
    }

    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text by removing dangerous patterns
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        try:
            for pattern, replacement in self.XSS_PATTERNS:
                text = pattern.sub(replacement, text)
            
            text = html.escape(text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text

        except Exception as e:
            logger.error(f"Error sanitizing text: {str(e)}")
            return text

    def sanitize_html(self, html_content: str) -> str:
        """
        Sanitize HTML content
        
        Args:
            html_content: HTML to sanitize
            
        Returns:
            Sanitized HTML
        """
        try:
            return bleach.clean(
                html_content,
                tags=self.ALLOWED_TAGS,
                attributes=self.ALLOWED_ATTRS,
                strip=True
            )

        except Exception as e:
            logger.error(f"Error sanitizing HTML: {str(e)}")
            return html_content
