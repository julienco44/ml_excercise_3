import sys
import os
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class Lemmatizer:
    def __init__(self, language: str = "en"):
        self.language = language
        self.nlp = None
        self._load_model()

    def _load_model(self):
        if not SPACY_AVAILABLE:
            print("spaCy not available. Lemmatization disabled.")
            return

        model_name = config.SPACY_MODELS.get(self.language)
        if not model_name:
            print(f"No spaCy model configured for language: {self.language}")
            return

        try:
            self.nlp = spacy.load(model_name, disable=["parser", "ner"])
            print(f"Loaded spaCy model: {model_name}")
        except OSError:
            print(f"spaCy model '{model_name}' not found.")
            print(f"Install with: python -m spacy download {model_name}")
            self.nlp = None

    def is_available(self) -> bool:
        return self.nlp is not None

    def lemmatize(self, text: str) -> List[str]:
        if not self.nlp or not text:
            return text.split() if text else []

        doc = self.nlp(text)
        return [token.lemma_.lower() for token in doc if token.is_alpha]

    def lemmatize_batch(self, texts: List[str], batch_size: int = 1000) -> List[List[str]]:
        if not self.nlp:
            return [text.split() for text in texts if text]

        results = []
        total = len(texts)
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            docs = self.nlp.pipe(batch)
            for doc in docs:
                lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
                results.append(lemmas)

            if (i + batch_size) % 10000 == 0:
                print(f"  Lemmatized {min(i + batch_size, total)}/{total} texts...")

        return results
