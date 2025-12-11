import logging
import re
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from config import config, logger

class SecurityChecker:
    """Performs security checks on inputs and requests"""

    def __init__(self):
        """Initialize the security checker"""
        self.blocked_patterns = [
            r'<script[^>]*>.*?</script>',
            r'<iframe[^>]*>.*?</iframe>',
            r'javascript:',
            r'vbscript:',
            r'eval\\s*\\(',
            r'exec\\s*\\(',
            r'system\\s*\\(',
            r'shell_exec\\s*\\(',
            r'file_get_contents\\s*\\(',
            r'include\\s*\\(',
            r'require\\s*\\('
        ]

        # Rate limiting storage (in production, use Redis)
        self.rate_limit_storage = {}

    def check_security_threats(self, text: str) -> Dict[str, Any]:
        """Check for security threats in text"""
        threats = []
        severity = 'low'

        try:
            # Check for XSS patterns
            if self._detect_xss(text):
                threats.append("Potential XSS attack detected")
                severity = 'high'

            # Check for SQL injection patterns
            if self._detect_sql_injection(text):
                threats.append("Potential SQL injection detected")
                severity = 'high'

            # Check for command injection patterns
            if self._detect_command_injection(text):
                threats.append("Potential command injection detected")
                severity = 'high'

            # Check for path traversal patterns
            if self._detect_path_traversal(text):
                threats.append("Potential path traversal detected")
                severity = 'medium'

            # Check for suspicious file extensions
            if self._detect_suspicious_files(text):
                threats.append("Suspicious file references detected")
                severity = 'medium'

            return {
                'status': 'success',
                'threats_detected': len(threats) > 0,
                'threats': threats,
                'severity': severity,
                'is_safe': len(threats) == 0
            }

        except Exception as e:
            logger.error(f"Error in security check: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'threats_detected': True,
                'threats': ['Security check failed'],
                'severity': 'high',
                'is_safe': False
            }

    def check_rate_limit(self, client_ip: str, endpoint: str) -> Dict[str, Any]:
        """Check if client has exceeded rate limit"""
        try:
            if not config.ENABLE_RATE_LIMITING:
                return {
                    'status': 'success',
                    'allowed': True,
                    'remaining': float('inf')
                }

            current_time = datetime.now()
            key = f"{client_ip}:{endpoint}"

            # Clean old entries
            self._clean_rate_limit_storage()

            # Check current rate
            if key in self.rate_limit_storage:
                requests = self.rate_limit_storage[key]
                # Remove old requests (older than 1 minute)
                requests = [req_time for req_time in requests if current_time - req_time < timedelta(minutes=1)]
                self.rate_limit_storage[key] = requests

                if len(requests) >= config.RATE_LIMIT_PER_MINUTE:
                    return {
                        'status': 'success',
                        'allowed': False,
                        'remaining': 0,
                        'reset_time': (requests[0] + timedelta(minutes=1)).isoformat()
                    }
            else:
                self.rate_limit_storage[key] = []

            # Add current request
            self.rate_limit_storage[key].append(current_time)

            remaining = config.RATE_LIMIT_PER_MINUTE - len(self.rate_limit_storage[key])

            return {
                'status': 'success',
                'allowed': True,
                'remaining': remaining
            }

        except Exception as e:
            logger.error(f"Error in rate limit check: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'allowed': True,  # Allow on error
                'remaining': float('inf')
            }

    def _detect_xss(self, text: str) -> bool:
        """Detect potential XSS attacks"""
        xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'<iframe[^>]*>.*?</iframe>',
            r'javascript:',
            r'vbscript:',
            r'onload\\s*=',
            r'onerror\\s*=',
            r'onclick\\s*='
        ]

        for pattern in xss_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _detect_sql_injection(self, text: str) -> bool:
        """Detect potential SQL injection attacks"""
        sql_patterns = [
            r'union\\s+select',
            r'drop\\s+table',
            r'delete\\s+from',
            r'insert\\s+into',
            r'update\\s+set',
            r'--',
            r'/\\*.*?\\*/',
            r'xp_cmdshell',
            r'sp_executesql'
        ]

        for pattern in sql_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _detect_command_injection(self, text: str) -> bool:
        """Detect potential command injection attacks"""
        cmd_patterns = [
            r';\\s*rm\\s+',
            r';\\s*cat\\s+',
            r';\\s*ls\\s+',
            r';\\s*whoami',
            r';\\s*id\\s*',
            r';\\s*ps\\s+',
            r';\\s*kill\\s+',
            r';\\s*chmod\\s+',
            r';\\s*chown\\s+',
            r';\\s*mkdir\\s+',
            r';\\s*rmdir\\s+'
        ]

        for pattern in cmd_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _detect_path_traversal(self, text: str) -> bool:
        """Detect potential path traversal attacks"""
        path_patterns = [
            r'\\.\\./',
            r'\\.\\.\\\\',
            r'\\.\\.%2f',
            r'\\.\\.%5c',
            r'\\.\\.%252f',
            r'\\.\\.%255c'
        ]

        for pattern in path_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _detect_suspicious_files(self, text: str) -> bool:
        """Detect suspicious file references"""
        suspicious_extensions = [
            r'\\.exe\\b',
            r'\\.bat\\b',
            r'\\.cmd\\b',
            r'\\.com\\b',
            r'\\.pif\\b',
            r'\\.scr\\b',
            r'\\.vbs\\b',
            r'\\.js\\b',
            r'\\.jar\\b',
            r'\\.php\\b',
            r'\\.asp\\b',
            r'\\.jsp\\b'
        ]

        for pattern in suspicious_extensions:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _clean_rate_limit_storage(self):
        """Clean old entries from rate limit storage"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=5)

        for key in list(self.rate_limit_storage.keys()):
            requests = self.rate_limit_storage[key]
            requests = [req_time for req_time in requests if req_time > cutoff_time]

            if requests:
                self.rate_limit_storage[key] = requests
            else:
                del self.rate_limit_storage[key]
