import re
from typing import Optional

try:
    from unidecode import unidecode
    UNIDECODE_AVAILABLE = True
except ImportError:
    UNIDECODE_AVAILABLE = False


class TextCleaner:
    def __init__(self):
        self.cleaning_regex = re.compile(
            r'[\d:;°"«»"„"€()\*+/–−§•…$&%~®™‹›‚\'▶︎–><\[\]{}@#^_\\|]+'
        )
        self.allowed_pattern = re.compile(r"^[a-zA-Z\s\-']+$")
        self.email_pattern = re.compile(r'\S+@\S+')
        self.url_pattern = re.compile(r'http\S+|www\.\S+')
        self.whitespace_pattern = re.compile(r'\s+')

    def clean(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None

        text = text.strip().strip('"')
        text = self.email_pattern.sub('', text)
        text = self.url_pattern.sub('', text)
        
        # Insert spaces around punctuation to ensure they are tokenized separately
        text = re.sub(r'([.,!?])', r' \1 ', text)
        
        text = self.cleaning_regex.sub(' ', text)

        if UNIDECODE_AVAILABLE:
            text = unidecode(text)

        text = self.whitespace_pattern.sub(' ', text).strip()
        text = text.lower()

        if not text:
            return None

        # Check if text contains at least one valid word (not just punctuation)
        has_words = any(c.isalnum() for c in text)
        if not has_words:
            return None

        return text

    def is_valid(self, text: str, min_length: int = 5) -> bool:
        if not text:
            return False
        words = text.split()
        return len(words) >= min_length
