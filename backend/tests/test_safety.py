import pytest

from agents.validation import (
    SecurityValidator,
    PIIValidator,
    SpamValidator,
    InputValidator,
    ContentSanitizer,
)
from agents.guardrails import MedicalGuardrails, ContentFilter


def test_security_validator_detects_xss():
    print("[test_safety] security_validator_detects_xss")
    validator = SecurityValidator()
    result = validator.validate("<script>alert(1)</script>")
    assert result["is_safe"] is False
    assert "XSS" in result["threats"][0]


def test_pii_validator_detects_and_sanitizes():
    print("[test_safety] pii_validator_detects_and_sanitizes")
    validator = PIIValidator()
    text = "Email me at a@b.com or call 555-123-4567"
    detected = validator.detect(text)
    assert detected["has_pii"] is True
    sanitized = validator.sanitize(text)
    assert "[EMAIL]" in sanitized and "[PHONE]" in sanitized


def test_spam_validator_flags_marketing():
    print("[test_safety] spam_validator_flags_marketing")
    validator = SpamValidator()
    result = validator.detect("Click here to buy now!!!")
    assert result["is_spam"] is True


def test_input_validator_format_checks():
    print("[test_safety] input_validator_format_checks")
    validator = InputValidator()
    long_text = "x" * (validator.max_length + 1)
    result = validator.validate_text(long_text)
    assert result.is_valid is False
    assert any("too long" in e.lower() for e in result.errors)


def test_content_sanitizer_strips_scripts():
    print("[test_safety] content_sanitizer_strips_scripts")
    sanitizer = ContentSanitizer()
    cleaned = sanitizer.sanitize_text("<script>bad</script>Hello")
    assert "bad" not in cleaned
    assert "Hello" in cleaned


def test_medical_guardrails_emergency_and_disclaimer():
    print("[test_safety] medical_guardrails_emergency_and_disclaimer")
    guardrails = MedicalGuardrails()
    safe, category, _ = guardrails.check_input("call 911 now chest pain")
    assert safe is False and category == "emergency"

    safe, category, _ = guardrails.check_input("Should I take ibuprofen?")
    assert safe is True and category == "disclaimer_required"


def test_content_filter_profanity(monkeypatch):
    print("[test_safety] content_filter_profanity")
    # Import module to patch global capability flag
    import agents.guardrails.content_filter as cf_mod

    filter_ = ContentFilter()
    # Force-enable profanity filter for this unit test, independent of global config.
    filter_.profanity_enabled = True
    # Simulate that profanity backend is available
    monkeypatch.setattr(cf_mod, "PROFANITY_AVAILABLE", True)
    monkeypatch.setattr(filter_, "_check_profanity", lambda text: True)
    result = filter_.filter("some bad words")
    assert result["is_clean"] is False
    assert "Profanity" in result["issues"][0]