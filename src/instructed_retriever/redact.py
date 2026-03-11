"""
Provides an MLflow span processor that redacts PII from span inputs and outputs
using Presidio-based redaction techniques.
"""

import json
import os
import re
from typing import Any

import usaddress  # type: ignore[import-untyped]
from mlflow.entities.span import Span
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_anonymizer import AnonymizerEngine

# ---------------------------------------------------------------------------
# Presidio engine setup
# ---------------------------------------------------------------------------

PII_SPACY_LANG_CODE = "en"
PII_SPACY_MODEL = os.getenv("PII_SPACY_MODEL", "en_core_web_md")
PII_ANALYZER: AnalyzerEngine | None = None

PII_ANONYMIZER = AnonymizerEngine()

# Words that should never be treated as PII
NO_REDACT_WORDS = [
    "role",
    "Role",
    "user",
    "system",
    "assistant",
    "tool",
    "Role.user",
    "Role.assistant",
    "Role.tool",
    "Role.system",
    "Unfrozen",
    "Frozen",
]

# Entity types to detect
PII_ENTITIES = [
    "CREDIT_CARD",
    "EMAIL_ADDRESS",
    "LOCATION",
    "PERSON",
    "PHONE_NUMBER",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_SSN",
    # "US_PASSPORT",
    "US_ITIN",
    "CREDIT_SCORE",
    "4_DIGIT_SSN",
    "DATE_OF_BIRTH",
]

# Dict keys whose values should not be redacted
REDACTION_KEYS_TO_SKIP = [
    "role",
    "tool_id",
    "context",
    "tools",
    "documents",
    "tool_calls",
    "deeplinks",
    "tool_call_id",
    "products",
    "self",
    "model_kwargs",
    "llm_model",
    "operation",
    "attribute_gateway",
    "analytics",
    "event_handler",
]

# ---------------------------------------------------------------------------
# Custom recognizers
# ---------------------------------------------------------------------------


def _initialize_pii_analyzer(pii_analyzer: AnalyzerEngine) -> AnalyzerEngine:
    """Register custom pattern recognizers for credit score, last-4 SSN, and DOB."""
    # Credit score (300–850)
    credit_score_pattern = PatternRecognizer(
        supported_entity="CREDIT_SCORE",
        patterns=[
            Pattern(
                name="credit_score_pattern",
                regex=r"\b(3[0-9]{2}|4[0-9]{2}|5[0-9]{2}|6[0-9]{2}|7[0-9]{2}|8[0-4][0-9]|850)\b(?!\w)",
                score=0.8,
            ),
        ],
    )
    # Last-4 digit SSN
    last4_ssn_pattern = PatternRecognizer(
        supported_entity="4_DIGIT_SSN",
        patterns=[
            Pattern(
                name="last_4_ssn_pattern",
                regex=r"\b\d{4}\b(?=\s|[.,!?;:]|\Z)",
                score=0.8,
            ),
        ],
    )
    # Date of birth
    dob_pattern = PatternRecognizer(
        supported_entity="DATE_OF_BIRTH",
        patterns=[
            Pattern(
                name="dob_pattern",
                regex=r"\b(\d{2}[-/\.]\d{2}[-/\.]\d{2,4}|(?:January|Jan|February|Feb"
                r"|March|Mar|April|Apr|May|June|Jun|July|Jul|August|Aug|September"
                r"|Sep|October|Oct|November|Nov|December|Dec) \d{1,2}(?:st|nd|rd"
                r"|th)?, \d{2,4})\b",
                score=0.8,
            ),
        ],
    )

    pii_analyzer.registry.add_recognizer(credit_score_pattern)
    pii_analyzer.registry.add_recognizer(last4_ssn_pattern)
    pii_analyzer.registry.add_recognizer(dob_pattern)
    return pii_analyzer


def _get_pii_analyzer() -> AnalyzerEngine:
    global PII_ANALYZER
    if PII_ANALYZER is None:
        PII_ANALYZER = _initialize_pii_analyzer(
            AnalyzerEngine(
                nlp_engine=SpacyNlpEngine(
                    models=[
                        {
                            "lang_code": PII_SPACY_LANG_CODE,
                            "model_name": PII_SPACY_MODEL,
                        },
                    ],
                ),
            )
        )
    return PII_ANALYZER


REDACTED_TAGS = [f"<{entity}>" for entity in PII_ENTITIES]
ADDRESS_NO_REDACT_WORDS = REDACTED_TAGS + NO_REDACT_WORDS

# ---------------------------------------------------------------------------
# Address redaction helpers
# ---------------------------------------------------------------------------


def fix_usaddress_dropping_chars(
    parsed_output: list[tuple[str, str]],
    original_json_string: str,
) -> list[tuple[str, str]]:
    fixed_output = []
    pointer = 0
    for word, tag in parsed_output:
        temp = ""
        while pointer < len(original_json_string) and not original_json_string[pointer:].startswith(
            word
        ):
            temp += original_json_string[pointer]
            pointer += 1
        fixed_word = temp + word
        fixed_output.append((fixed_word, tag))
        pointer += len(word)
    if pointer < len(original_json_string):
        fixed_output.append((original_json_string[pointer:], "Recipient"))
    return fixed_output


def find_and_replace_addresses(s: str) -> str:
    parsed_addr = fix_usaddress_dropping_chars(usaddress.parse(s), s)
    output = []
    non_address_tags = ("Recipient", "NotAddress", "BuildingName")
    for word, tag in parsed_addr:
        if tag in non_address_tags or any(
            special_word in word for special_word in ADDRESS_NO_REDACT_WORDS
        ):
            output.append(word)
        else:
            output.append("<ADDRESS_PART>")
    return "".join(output)


# ---------------------------------------------------------------------------
# Core redaction functions
# ---------------------------------------------------------------------------


def redact_numerics_longer_than(length: int, content: str) -> str:
    return re.sub(rf"\b\d{{{length},}}\b", "<NUMERIC>", content)


def presidio_model_anonymize(content: str) -> str:
    """Run Presidio analysis + anonymization, then address & numeric redaction."""
    if content.strip() == "":
        return content

    analyzer_result = _get_pii_analyzer().analyze(
        text=content,
        language="en",
        entities=PII_ENTITIES,
        allow_list=NO_REDACT_WORDS,
    )
    anonymized_result = PII_ANONYMIZER.anonymize(
        text=content,
        analyzer_results=analyzer_result,  # type: ignore[arg-type]
    )
    address_redacted_content = find_and_replace_addresses(anonymized_result.text)
    return redact_numerics_longer_than(2, address_redacted_content)


def recursive_redact_json(obj: Any) -> Any:
    """Recursively redact PII from a JSON-like object (dict / list / str)."""
    if isinstance(obj, str):
        return presidio_model_anonymize(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in REDACTION_KEYS_TO_SKIP:
                obj[k] = recursive_redact_json(v)
        return obj
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = recursive_redact_json(v)
        return obj
    return obj


def redact_pii(content: Any) -> Any:
    """Redact PII from lists, dicts, and JSON / plain strings."""
    if isinstance(content, list | dict):
        return recursive_redact_json(content)
    if isinstance(content, str):
        try:
            json_obj = json.loads(content)
            return json.dumps(recursive_redact_json(json_obj))
        except json.decoder.JSONDecodeError:
            return presidio_model_anonymize(content)
    return content


# ---------------------------------------------------------------------------
# MLflow span processor
# ---------------------------------------------------------------------------


def redact_span_pii(span: Span) -> None:
    """MLflow span processor — redacts PII from span inputs and outputs in-place."""
    inputs = span.inputs
    if inputs and isinstance(inputs, dict):
        span.set_inputs(recursive_redact_json(inputs))  #  type: ignore[attr-defined]

    outputs = span.outputs
    if outputs is not None:
        if isinstance(outputs, dict):
            span.set_outputs(recursive_redact_json(outputs))  #  type: ignore[attr-defined]
        elif isinstance(outputs, str):
            span.set_outputs(redact_pii(outputs))  #  type: ignore[attr-defined]
        elif isinstance(outputs, list):
            span.set_outputs(redact_pii(outputs))  #  type: ignore[attr-defined]
