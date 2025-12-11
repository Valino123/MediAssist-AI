import logging
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, validator

from config import config, logger

class InputValidationResult(BaseModel):
    """Result of input validation"""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    sanitized_input: Optional[str] = None

class InputValidator:
    """Validates and sanitizes user inputs"""

    def __init__(self):
        """Initialize the input validator"""
        self.max_length = config.MAX_MESSAGE_LENGTH
        self.profanity_enabled = config.ENABLE_PROFANITY_FILTER

        # Compile regex patterns
        self.email_pattern = re.compile(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b')
        self.phone_pattern = re.compile(r'\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\\\(\\\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

        # Medical-specific patterns
        self.medical_id_pattern = re.compile(r'\\b[A-Z]{2,3}\\d{6,8}\\b')  # Medical record IDs
        self.drug_pattern = re.compile(r'\\b[A-Z][a-z]+(?:ine|ol|ide|ate)\\b')  # Drug names

    def validate_text_input(self, text: str) -> InputValidationResult:
        """Validate text input for chat messages"""
        errors = []
        warnings = []

        try:
            # Check length
            if len(text) > self.max_length:
                errors.append(f"Message too long. Maximum {self.max_length} characters allowed.")

            # Check for empty input
            if not text.strip():
                errors.append("Message cannot be empty.")

            # Check for excessive whitespace
            if len(text) - len(text.strip()) > 100:
                warnings.append("Message contains excessive whitespace.")

            # Check for repeated characters
            if self._has_excessive_repetition(text):
                warnings.append("Message contains excessive character repetition.")

            # Check for potential spam patterns
            if self._detect_spam_patterns(text):
                errors.append("Message appears to be spam.")

            # Check for personal information
            personal_info = self._detect_personal_info(text)
            if personal_info:
                warnings.append(f"Message may contain personal information: {', '.join(personal_info)}")

            # Check for medical record information
            medical_info = self._detect_medical_info(text)
            if medical_info:
                warnings.append(f"Message may contain medical record information: {', '.join(medical_info)}")

            return InputValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                sanitized_input=text.strip() if len(errors) == 0 else None
            )

        except Exception as e:
            logger.error(f"Error in text validation: {str(e)}")
            return InputValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                sanitized_input=None
            )

    def validate_image_input(self, image_data: str, file_size: int) -> InputValidationResult:
        """Validate image input"""
        errors = []
        warnings = []

        try:
            # Check file size
            if file_size > config.MAX_FILE_SIZE:
                errors.append(f"Image too large. Maximum {config.MAX_FILE_SIZE / 1024 / 1024:.1f}MB allowed.")

            # Check for valid base64
            if not self._is_valid_base64_image(image_data):
                errors.append("Invalid image format.")

            # Check for suspicious patterns in base64
            if self._detect_suspicious_image_patterns(image_data):
                warnings.append("Image may contain suspicious content.")

            return InputValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                sanitized_input=image_data if len(errors) == 0 else None
            )

        except Exception as e:
            logger.error(f"Error in image validation: {str(e)}")
            return InputValidationResult(
                is_valid=False,
                errors=[f"Image validation error: {str(e)}"],
                warnings=[],
                sanitized_input=None
            )

    def _has_excessive_repetition(self, text: str) -> bool:
        """Check for excessive character repetition"""
        # Check for repeated characters (more than 5 in a row)
        return bool(re.search(r'(.)\\1{5,}', text))

    def _detect_spam_patterns(self, text: str) -> bool:
        """Detect potential spam patterns"""
        # Check for excessive capitalization
        if len(re.findall(r'[A-Z]', text)) / len(text) > 0.7:
            return True

        # Check for excessive punctuation
        if len(re.findall(r'[!?]{3,}', text)) > 0:
            return True

        # Check for repeated words
        words = text.lower().split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                if word_counts[word] > len(words) * 0.3:
                    return True

        return False

    def _detect_personal_info(self, text: str) -> List[str]:
        """Detect potential personal information"""
        detected = []

        # Check for email addresses
        if self.email_pattern.search(text):
            detected.append("email addresses")

        # Check for phone numbers
        if self.phone_pattern.search(text):
            detected.append("phone numbers")

        # Check for URLs
        if self.url_pattern.search(text):
            detected.append("URLs")

        return detected

    def _detect_medical_info(self, text: str) -> List[str]:
        """Detect potential medical record information"""
        detected = []

        # Check for medical record IDs
        if self.medical_id_pattern.search(text):
            detected.append("medical record IDs")

        # Check for drug names
        if self.drug_pattern.search(text):
            detected.append("drug names")

        return detected

    def _is_valid_base64_image(self, image_data: str) -> bool:
        """Check if base64 string is a valid image"""
        try:
            import base64
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]

            # Decode base64
            decoded = base64.b64decode(image_data)

            # Check if it's a valid image by trying to open it
            from PIL import Image
            import io
            Image.open(io.BytesIO(decoded))
            return True

        except Exception:
            return False

    def _detect_suspicious_image_patterns(self, image_data: str) -> bool:
        """Detect suspicious patterns in image data"""
        # Check for extremely large base64 strings
        if len(image_data) > 10 * 1024 * 1024:  # 10MB
            return True

        # Check for unusual character patterns
        if len(set(image_data)) < 10:  # Very low character diversity
            return True

        return False
