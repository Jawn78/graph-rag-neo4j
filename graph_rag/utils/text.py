"""
Shared text sanitization utilities.

Pre-compiles regex patterns at module load time for performance.
All text sanitization should go through this module to ensure consistency
and avoid redundant processing.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for sanitization (compiled once at module load)
_CTRL_CHARS = re.compile(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]')
_ZERO_WIDTH = re.compile(r'[\u200b-\u200d\ufeff]')
_LINE_SEP = re.compile(r'[\u2028\u2029]')
_NBSP = re.compile(r'[\u00a0]')
_REPEATED_CHARS = re.compile(r'(.)\1{10,}')
_EXCESSIVE_REPEATS = re.compile(r'(.)\1{20,}')
_WHITESPACE_NORMALIZE = re.compile(r'[ \t]+')
_NEWLINE_NORMALIZE = re.compile(r'\n+')

# Sentence splitting for chunking
_SENTENCE_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')


def sanitize_text(text: str, check_corruption: bool = True) -> str:
    """
    Sanitize text by removing control characters and normalizing Unicode.

    This is the single source of truth for text sanitization. All other modules
    should call this function rather than implementing their own sanitization.

    Args:
        text: The text to sanitize
        check_corruption: If True, return empty string for corrupted text

    Returns:
        Sanitized text, or empty string if text is corrupted/empty
    """
    if not text:
        return ""

    # Check for corruption first (if requested)
    if check_corruption and is_corrupted(text):
        logger.warning(f"Skipping corrupted text: {text[:50]}...")
        return ""

    # Apply sanitization in order (most common issues first)
    text = _CTRL_CHARS.sub(' ', text)        # Control characters
    text = _ZERO_WIDTH.sub('', text)         # Zero-width characters
    text = _LINE_SEP.sub('\n', text)         # Line/paragraph separators
    text = _NBSP.sub(' ', text)              # Non-breaking space
    text = _REPEATED_CHARS.sub(r'\1', text)  # Excessive repetition
    text = _WHITESPACE_NORMALIZE.sub(' ', text)
    text = _NEWLINE_NORMALIZE.sub('\n', text)

    # Ensure proper UTF-8 encoding
    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception:
        pass

    return text.strip()


def is_corrupted(text: str) -> bool:
    """
    Check if text appears to be corrupted or garbled.

    Detects:
    - Excessive repeated characters (e.g., "aaaaaaaaaaaaaaaaaaaaa")
    - Low ratio of printable characters
    - High ratio of control characters

    Args:
        text: The text to check

    Returns:
        True if text appears corrupted
    """
    if not text or len(text) < 10:
        return False

    # Check for excessive repeated characters (20+)
    if _EXCESSIVE_REPEATS.search(text):
        return True

    # Check printability ratio
    printable_count = sum(1 for c in text if c.isprintable() or c.isspace())
    if printable_count / len(text) < 0.7:
        return True

    # Check control character ratio
    control_count = sum(1 for c in text if ord(c) < 32 and c not in '\t\n\r')
    if control_count / len(text) > 0.1:
        return True

    return False


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences for chunking.

    Args:
        text: The text to split

    Returns:
        List of sentences
    """
    return _SENTENCE_SPLIT.split((text or "").strip())


def sanitize_answer(text: str) -> str:
    """
    Light sanitization for LLM-generated answers.

    Answers from the LLM should already be clean, so we only apply
    minimal sanitization to handle potential Unicode issues.

    Args:
        text: The answer text

    Returns:
        Sanitized answer
    """
    if not text:
        return ""

    text = _ZERO_WIDTH.sub('', text)
    text = _LINE_SEP.sub('\n', text)
    text = _NBSP.sub(' ', text)
    text = _NEWLINE_NORMALIZE.sub('\n', text)
    text = _WHITESPACE_NORMALIZE.sub(' ', text)

    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception:
        pass

    return text.strip()
