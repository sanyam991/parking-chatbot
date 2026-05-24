"""
Guardrails Module - Data Protection and Safety Filtering.

This module implements two types of protection:

1. INPUT GUARDRAILS (check_input):
   - Detects prompt injection attempts (users trying to override system instructions)
   - Blocks requests for other users' personal information
   - Prevents malicious queries

2. OUTPUT GUARDRAILS (filter_output):
   - Scans LLM responses for accidentally leaked PII
   - Redacts any sensitive information (phone numbers, emails, card numbers)
   - Ensures no internal system details are exposed

HOW PII DETECTION WORKS:
- Uses Microsoft Presidio, which combines:
  * Named Entity Recognition (NER) - ML models trained to find names, orgs, etc.
  * Pattern matching - Regex for structured data (phone, email, SSN)
  * Context analysis - Checks surrounding words (e.g., "call me at" before a number)

- Each detection has a confidence score (0.0 to 1.0)
- We only act on detections above our threshold (default: 0.7)

ALTERNATIVE APPROACH (fallback if Presidio is not available):
- Simple regex-based detection for common PII patterns
- Less accurate but works without extra model downloads

WHY ALLOWLISTING IS NEEDED:
Presidio and regex patterns cannot distinguish between a *user's* private phone number
and a *business's* public support line.  Without an allowlist, official ParkSmart contacts
like "+91-40-9999-0000" or "support@parksmart.in" would be redacted from bot responses,
breaking the user experience.

MASKING DECISION LOGIC:
  PRIVATE_USER_INFO   → always redacted  (user emails, user phone numbers, payment info)
  PUBLIC_BUSINESS_INFO → never redacted  (company contacts, official addresses, websites)

The filter uses a *placeholder-substitution* strategy for PUBLIC_BUSINESS_INFO:
  1. Replace every known public contact with a unique sentinel token  (e.g. __PS_0__)
  2. Run Presidio / regex on the sentinel-substituted text
  3. Restore sentinels back to the original public values

This guarantees public contacts are untouched regardless of how Presidio tokenises them.
"""

import re
from typing import Any, Dict, List, Tuple

from config.settings import settings

# ---------------------------------------------------------------------------
# PUBLIC_BUSINESS_INFO – official ParkSmart contacts that must NEVER be masked
# ---------------------------------------------------------------------------
# These are sourced from parking_info.txt and are intentionally public.
# Add any new official contacts here to protect them automatically.
PUBLIC_BUSINESS_INFO: Tuple[str, ...] = (
    # ── Phone numbers ──────────────────────────────────────────────────────
    "+91 7985819872",
    "+91-7985819872",
    "+91-40-5555-7788",
    "+91-98765-43210",
    "+91-40-9999-0000",
    # ── Email addresses ────────────────────────────────────────────────────
    "support@parksmart.in",
    "reservations@parksmart.in",
    "corporate@parksmart.in",
    # ── Website ────────────────────────────────────────────────────────────
    "www.parksmart.in",
    "parksmart.in",
)

# Sentinel template used during placeholder substitution.
# Unlikely to appear in any real LLM response.
_SENTINEL_TPL = "__PARKSMART_CONTACT_{idx}__"


class Guardrails:
    """
    Guardrails for protecting sensitive data and preventing misuse.

    Uses a dual approach:
    - Presidio NLP analyzer (if available) for high-accuracy PII detection
    - Regex fallback patterns for basic protection

    Public business contacts listed in PUBLIC_BUSINESS_INFO are *never* redacted;
    only genuine private user data is masked.
    """

    def __init__(self):
        """Initialize the guardrails with PII detection capabilities."""
        self.enabled = settings.guardrails_enabled
        self.confidence_threshold = settings.pii_confidence_threshold

        # Presidio engines are loaded lazily on first filter_output() call
        # to avoid blocking startup with spaCy model loading (~30-60s).
        self.presidio_available = False
        self._presidio_initialized = False
        self.analyzer = None
        self.anonymizer = None

        # Keywords that suggest a prompt injection attempt
        self.injection_patterns = [
            r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|rules|prompts)",
            r"ignore\s+(previous|all|above)\s+(instructions|rules|prompts)",
            r"forget\s+(everything|all|your)\s*(instructions|rules|training)?",
            r"you\s+are\s+now\s+(?!parksmart)",  # "you are now DAN/evil" etc.
            r"override\s+(system|safety|your)\s*(prompt|rules|instructions)?",
            r"pretend\s+you\s+(are|don't\s+have)",
            r"reveal\s+(your|the)\s+(system|initial|original)\s*(prompt|instructions)?",
            r"show\s+me\s+(your|the)\s+(system|initial)\s*prompt",
            r"what\s+(is|are)\s+your\s+(system|initial)\s*(prompt|instructions)",
        ]

        # Keywords suggesting request for other users' data
        self.data_request_patterns = [
            r"show\s+me\s+(all\s+)?(other\s+)?(users?'?s?|customers?'?s?)\s*(reservations?|bookings?|data|info)?",
            r"show\s+me\s+all\s+other\s+",
            r"(list|give|tell)\s+me\s+.*(other|all)\s*(people|users?|customers?)",
            r"list\s+all\s+(customers?|users?|bookings?|reservations?)",
            r"who\s+(else\s+)?(has|made|booked)\s+",
            r"(database|db|table|record)\s*(dump|export|contents?|data)",
            r"(admin|administrator|internal)\s*(password|access|credentials?)",
        ]

        # Regex patterns for PRIVATE_USER_INFO detection (Presidio fallback).
        # These are intentionally narrow to avoid false-positives on business info.
        self.pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"(?<!\d)(\+?(\d[\s\-.]?){9,14}\d)(?!\d)",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "ssn": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        }

        # Regex patterns that are always safe regardless of content
        # (e.g. masked user emails produced by masking.py – already anonymised)
        self.safe_patterns = [
            r"[A-Za-z0-9]{1,2}\*{2,}@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",  # sa****@gmail.com
        ]

    # -----------------------------------------------------------------------
    # Lazy Presidio initialisation
    # -----------------------------------------------------------------------

    def _ensure_presidio(self):
        """Load Presidio engines on first use (avoids blocking startup)."""
        if self._presidio_initialized:
            return
        self._presidio_initialized = True
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            self.presidio_available = True
            print("✓ Presidio PII analyzer loaded successfully.")
        except (ImportError, Exception) as e:
            print(f"⚠ Presidio not available, using regex fallback. Reason: {e}")

    # -----------------------------------------------------------------------
    # Public-info protection helpers (placeholder substitution)
    # -----------------------------------------------------------------------

    def _protect_public_info(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Temporarily replace every PUBLIC_BUSINESS_INFO value with a unique
        sentinel token so that Presidio / regex never sees them.

        Returns:
            protected_text: text with sentinels in place of public contacts.
            restore_map:    mapping sentinel → original value for restoration.

        WHY: Presidio cannot distinguish between a user's private phone number
        and an official business support line.  By substituting known-safe values
        before analysis, we guarantee they are never redacted regardless of how
        the NER model tokenises the surrounding context.
        """
        restore_map: Dict[str, str] = {}
        protected = text
        # Sort longest-first so that "+91-7985819872" is matched before "7985819872"
        for idx, public_value in enumerate(sorted(PUBLIC_BUSINESS_INFO, key=len, reverse=True)):
            sentinel = _SENTINEL_TPL.format(idx=idx)
            if public_value in protected:
                protected = protected.replace(public_value, sentinel)
                restore_map[sentinel] = public_value
        return protected, restore_map

    def _restore_public_info(self, text: str, restore_map: Dict[str, str]) -> str:
        """Reverse _protect_public_info — swap sentinels back to original values."""
        restored = text
        for sentinel, original in restore_map.items():
            restored = restored.replace(sentinel, original)
        return restored

    # -----------------------------------------------------------------------
    # Input guardrail
    # -----------------------------------------------------------------------

    def check_input(self, user_input: str) -> Dict[str, Any]:
        """
        Check user input for malicious content.

        Checks for:
        1. Prompt injection attempts
        2. Requests for other users' private data

        Args:
            user_input: The raw user message

        Returns:
            Dict with:
            - blocked: bool (True if message should be blocked)
            - reason: str (why it was blocked, if applicable)
            - message: str (response to send to user if blocked)
        """
        if not self.enabled:
            return {"blocked": False, "reason": None, "message": None}

        # Ensure input is a plain string
        user_input = str(user_input)
        input_lower = user_input.lower()

        # Check for prompt injection
        for pattern in self.injection_patterns:
            if re.search(pattern, input_lower):
                return {
                    "blocked": True,
                    "reason": "prompt_injection",
                    "message": (
                        "I'm sorry, but I can only help with parking-related queries "
                        "such as information about our facility, pricing, availability, "
                        "and making reservations. How can I assist you today?"
                    ),
                }

        # Check for requests for other users' data
        for pattern in self.data_request_patterns:
            if re.search(pattern, input_lower):
                return {
                    "blocked": True,
                    "reason": "data_privacy",
                    "message": (
                        "I'm sorry, I cannot share other users' personal information "
                        "or reservation details. I can only help you with your own "
                        "reservation or provide general parking information. "
                        "How else can I help you?"
                    ),
                }

        return {"blocked": False, "reason": None, "message": None}

    def filter_output(self, response: str) -> str:
        """
        Filter the LLM output to remove any accidentally leaked PRIVATE_USER_INFO.

        PUBLIC_BUSINESS_INFO (official ParkSmart contacts) is never touched.

        Strategy:
          1. Protect public contacts via placeholder substitution.
          2. Also collect ranges of already-masked user emails (sa****@gmail.com)
             so the regex fallback doesn't double-redact them.
          3. Run Presidio or regex on the substituted text.
          4. Restore public contacts from sentinels.

        Args:
            response: The raw LLM response.

        Returns:
            Filtered response with only PRIVATE_USER_INFO redacted.
        """
        if not self.enabled:
            return response

        # Lazy-load Presidio on first real use (spaCy model can take 30-60s).
        self._ensure_presidio()

        response = str(response) if not isinstance(response, str) else response

        # Step 1 – protect public business contacts with sentinel placeholders.
        protected, restore_map = self._protect_public_info(response)

        # Step 2 – also find already-masked user emails (safe_patterns) so the
        # fallback regex doesn't accidentally re-redact them.
        safe_ranges: List[tuple] = []
        for pattern in self.safe_patterns:
            for m in re.finditer(pattern, protected):
                safe_ranges.append((m.start(), m.end()))

        # Step 3 – run PII detection on the sentinel-substituted text.
        if self.presidio_available:
            filtered = self._filter_with_presidio(protected, safe_ranges)
        else:
            filtered = self._filter_with_regex(protected, safe_ranges)

        # Step 4 – restore public contacts (sentinels → original values).
        return self._restore_public_info(filtered, restore_map)

    def _filter_with_presidio(self, text: str, safe_ranges: List[tuple]) -> str:
        """
        Use Presidio NLP analyzer to detect and redact PII.

        Presidio detects: names, emails, phones, credit cards,
        addresses, SSNs, and many more entity types.
        """
        from presidio_analyzer import AnalyzerEngine

        # Analyze the text for PII entities
        results = self.analyzer.analyze(
            text=text,
            language="en",
            entities=[
                "PHONE_NUMBER",
                "EMAIL_ADDRESS",
                "CREDIT_CARD",
                "US_SSN",
                "IBAN_CODE",
                "IP_ADDRESS",
            ],
            score_threshold=self.confidence_threshold,
        )

        # Filter out detections that fall within "safe" ranges
        filtered_results = []
        for result in results:
            is_safe = any(safe_start <= result.start and result.end <= safe_end for safe_start, safe_end in safe_ranges)
            if not is_safe:
                filtered_results.append(result)

        # If no PII found, return as-is
        if not filtered_results:
            return text

        # Redact detected PII
        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=filtered_results,
        )

        return anonymized.text

    def _filter_with_regex(self, text: str, safe_ranges: List[tuple]) -> str:
        """
        Fallback: Use regex patterns to detect and redact PII.
        Less accurate than Presidio but works without extra dependencies.
        """
        filtered_text = text

        for pii_type, pattern in self.pii_patterns.items():
            for match in re.finditer(pattern, filtered_text):
                # Check if this match is in a safe range
                is_safe = any(
                    safe_start <= match.start() and match.end() <= safe_end for safe_start, safe_end in safe_ranges
                )

                if not is_safe:
                    # Replace with redaction marker
                    redacted = f"[{pii_type.upper()}_REDACTED]"
                    filtered_text = filtered_text[: match.start()] + redacted + filtered_text[match.end() :]
                    # Recalculate safe_ranges offset (text length changed)
                    break  # Start over to handle offset changes

        return filtered_text

    def detect_pii_in_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PRIVATE_USER_INFO entities in text (for evaluation/debugging).

        PUBLIC_BUSINESS_INFO values are excluded from results — they are
        intentionally public and should never appear as PII detections.

        Returns a list of detected entities with their types and positions.
        """
        # Protect public contacts first so they don't show as PII detections.
        protected, _ = self._protect_public_info(text)

        if self.presidio_available:
            results = self.analyzer.analyze(
                text=protected,
                language="en",
                score_threshold=self.confidence_threshold,
            )
            return [
                {
                    "entity_type": r.entity_type,
                    "start": r.start,
                    "end": r.end,
                    "score": r.score,
                    "text": protected[r.start : r.end],
                }
                for r in results
            ]
        else:
            detections = []
            for pii_type, pattern in self.pii_patterns.items():
                for match in re.finditer(pattern, protected):
                    detections.append(
                        {
                            "entity_type": pii_type.upper(),
                            "start": match.start(),
                            "end": match.end(),
                            "score": 1.0,
                            "text": match.group(),
                        }
                    )
            return detections
