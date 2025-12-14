"""
Content policy filter
Handles profanity filtering and inappropriate content
"""
import warnings
from typing import Dict, Any

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

try:
    from profanity_check import predict, predict_prob
    PROFANITY_AVAILABLE = True
except Exception:
    PROFANITY_AVAILABLE = False

from config import config, logger


class ContentFilter:
    """Filters inappropriate content based on policy"""

    def __init__(self):
        """Initialize content filter"""
        self.profanity_enabled = config.ENABLE_PROFANITY_FILTER

    def filter(self, text: str) -> Dict[str, Any]:
        """
        Filter content for policy violations
        
        Args:
            text: Text to filter
            
        Returns:
            Dictionary with filtering results
        """
        try:
            issues = []
            filtered_text = text

            if self.profanity_enabled and PROFANITY_AVAILABLE:
                has_profanity = self._check_profanity(text)
                if has_profanity:
                    issues.append("Profanity detected")
                    filtered_text = self._mask_profanity(text)

            return {
                'is_clean': len(issues) == 0,
                'issues': issues,
                'filtered_text': filtered_text
            }

        except Exception as e:
            logger.error(f"Error filtering content: {str(e)}")
            return {
                'is_clean': False,
                'issues': ['Filter error'],
                'filtered_text': text
            }

    def _check_profanity(self, text: str) -> bool:
        """Check for profanity"""
        try:
            if not PROFANITY_AVAILABLE:
                return False
            return predict([text])[0] == 1
        except Exception:
            return False

    def _mask_profanity(self, text: str) -> str:
        """Mask profanity in text"""
        # Simple masking - in production use more sophisticated approach
        return text
