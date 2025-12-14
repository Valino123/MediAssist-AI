"""
Input format validation module
Validates message format, length, and structure
"""
import re
import base64
import io
from typing import Dict, Any
from pydantic import BaseModel

from config import config, logger


class InputValidationResult(BaseModel):
    """Result of input validation"""
    is_valid: bool
    errors: list = []
    warnings: list = []


class InputValidator:
    """Validates input format and structure"""

    def __init__(self):
        """Initialize validator"""
        self.max_length = config.MAX_MESSAGE_LENGTH

    def validate_text(self, text: str) -> InputValidationResult:
        """
        Validate text input format
        
        Args:
            text: Text to validate
            
        Returns:
            InputValidationResult with validation status
        """
        errors = []
        warnings = []

        try:
            if not text or not text.strip():
                errors.append("Message cannot be empty")

            if len(text) > self.max_length:
                errors.append(f"Message too long. Maximum {self.max_length} characters")

            if len(text) - len(text.strip()) > 100:
                warnings.append("Excessive whitespace detected")

            if re.search(r'(.)\1{5,}', text):
                warnings.append("Excessive character repetition")

            return InputValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings
            )

        except Exception as e:
            logger.error(f"Error validating text: {str(e)}")
            return InputValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"]
            )

    def validate_image(self, image_data: str, file_size: int) -> InputValidationResult:
        """
        Validate image input
        
        Args:
            image_data: Base64 encoded image
            file_size: Size of image file
            
        Returns:
            InputValidationResult with validation status
        """
        errors = []

        try:
            if file_size > config.MAX_FILE_SIZE:
                errors.append(f"Image too large. Maximum {config.MAX_FILE_SIZE / 1024 / 1024:.1f}MB")

            if not self._is_valid_base64_image(image_data):
                errors.append("Invalid image format")

            return InputValidationResult(
                is_valid=len(errors) == 0,
                errors=errors
            )

        except Exception as e:
            logger.error(f"Error validating image: {str(e)}")
            return InputValidationResult(
                is_valid=False,
                errors=[f"Image validation error: {str(e)}"]
            )

    def _is_valid_base64_image(self, image_data: str) -> bool:
        """Check if base64 string is valid image"""
        try:
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            decoded = base64.b64decode(image_data)
            
            from PIL import Image
            Image.open(io.BytesIO(decoded))
            return True

        except Exception:
            return False
