"""
Security validation module
Handles XSS, SQL injection, command injection, and path traversal detection
"""
import re
from typing import Dict, Any, List
from datetime import datetime, timedelta

from config import config, logger


class SecurityValidator:
    """Validates input for security threats"""
    
    # XSS patterns
    XSS_PATTERNS = [
        re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
        re.compile(r'<iframe[^>]*>.*?</iframe>', re.IGNORECASE | re.DOTALL),
        re.compile(r'javascript:', re.IGNORECASE),
        re.compile(r'vbscript:', re.IGNORECASE),
        re.compile(r'onload\s*=', re.IGNORECASE),
        re.compile(r'onerror\s*=', re.IGNORECASE),
        re.compile(r'onclick\s*=', re.IGNORECASE),
    ]
    
    # SQL injection patterns
    SQL_PATTERNS = [
        re.compile(r'union\s+select', re.IGNORECASE),
        re.compile(r'drop\s+table', re.IGNORECASE),
        re.compile(r'delete\s+from', re.IGNORECASE),
        re.compile(r'insert\s+into', re.IGNORECASE),
        re.compile(r'update\s+set', re.IGNORECASE),
        re.compile(r'--', re.IGNORECASE),
    ]
    
    # Command injection patterns
    CMD_PATTERNS = [
        re.compile(r';\s*rm\s+', re.IGNORECASE),
        re.compile(r';\s*cat\s+', re.IGNORECASE),
        re.compile(r';\s*ls\s+', re.IGNORECASE),
    ]
    
    # Path traversal patterns
    PATH_PATTERNS = [
        re.compile(r'\.\.[/\\]', re.IGNORECASE),
        re.compile(r'\.\.%2f|\.\.%5c', re.IGNORECASE),
    ]

    def __init__(self):
        """Initialize security validator"""
        self.rate_limit_storage = {}

    def validate(self, text: str) -> Dict[str, Any]:
        """
        Validate text for security threats
        
        Args:
            text: Text to validate
            
        Returns:
            Dictionary with validation results and threat details
        """
        threats = []
        severity = 'none'

        try:
            if self._has_xss(text):
                threats.append("XSS attack detected")
                severity = 'high'

            if self._has_sql_injection(text):
                threats.append("SQL injection detected")
                severity = 'high'

            if self._has_command_injection(text):
                threats.append("Command injection detected")
                severity = 'high'

            if self._has_path_traversal(text):
                threats.append("Path traversal detected")
                severity = 'medium'

            return {
                'is_safe': len(threats) == 0,
                'threats': threats,
                'severity': severity
            }

        except Exception as e:
            logger.error(f"Error in security validation: {str(e)}")
            return {
                'is_safe': False,
                'threats': ['Validation error'],
                'severity': 'high'
            }

    def _has_xss(self, text: str) -> bool:
        """Check for XSS patterns"""
        return any(pattern.search(text) for pattern in self.XSS_PATTERNS)

    def _has_sql_injection(self, text: str) -> bool:
        """Check for SQL injection patterns"""
        return any(pattern.search(text) for pattern in self.SQL_PATTERNS)

    def _has_command_injection(self, text: str) -> bool:
        """Check for command injection patterns"""
        return any(pattern.search(text) for pattern in self.CMD_PATTERNS)

    def _has_path_traversal(self, text: str) -> bool:
        """Check for path traversal patterns"""
        return any(pattern.search(text) for pattern in self.PATH_PATTERNS)

    def check_rate_limit(self, client_ip: str, endpoint: str) -> Dict[str, Any]:
        """Check if client has exceeded rate limit"""
        try:
            if not config.ENABLE_RATE_LIMITING:
                return {'allowed': True, 'remaining': float('inf')}

            current_time = datetime.now()
            key = f"{client_ip}:{endpoint}"

            self._clean_rate_limit_storage()

            if key in self.rate_limit_storage:
                requests = self.rate_limit_storage[key]
                requests = [t for t in requests if current_time - t < timedelta(minutes=1)]
                self.rate_limit_storage[key] = requests

                if len(requests) >= config.RATE_LIMIT_PER_MINUTE:
                    return {
                        'allowed': False,
                        'remaining': 0,
                        'reset_time': (requests[0] + timedelta(minutes=1)).isoformat()
                    }
            else:
                self.rate_limit_storage[key] = []

            self.rate_limit_storage[key].append(current_time)
            remaining = config.RATE_LIMIT_PER_MINUTE - len(self.rate_limit_storage[key])

            return {'allowed': True, 'remaining': remaining}

        except Exception as e:
            logger.error(f"Error in rate limit check: {str(e)}")
            return {'allowed': True, 'remaining': float('inf')}

    def _clean_rate_limit_storage(self):
        """Clean old entries from rate limit storage"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=5)

        for key in list(self.rate_limit_storage.keys()):
            requests = [t for t in self.rate_limit_storage[key] if t > cutoff_time]
            if requests:
                self.rate_limit_storage[key] = requests
            else:
                del self.rate_limit_storage[key]

