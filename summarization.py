"""Extractive and optional transformer-based summarization."""
import re
from functools import lru_cache


def lexrank_summary(text: str, sentences_count: int = 5) -> str:
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.summarizers.lex_rank import LexRankSummarizer
    if not text.strip():
        return ''
    parser = PlaintextParser.from_string(text, Tokenizer('english'))
    return ' '.join(str(s) for s in LexRankSummarizer()(parser.document, sentences_count))


def chunk_text(text: str, max_words: int = 350, overlap_words: int = 40) -> list[str]:
    """Word-bounded chunks with overlap; transformer also truncates by tokens."""
    if max_words <= 0 or not 0 <= overlap_words < max_words:
        raise ValueError('Require max_words > overlap_words >= 0')
    words = text.split()
    step = max_words - overlap_words
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), step)
            if words[i:i + max_words] and (i == 0 or i < len(words))]


@lru_cache(maxsize=2)
def _pipeline(model_name: str):
    from transformers import pipeline
    return pipeline('summarization', model=model_name, device=-1)


def transformer_summary(text: str, model_name: str = 't5-small', max_chunks: int = 12) -> str:
    """Demo inference; model downloads may be large and output needs human verification."""
    if not text.strip():
        return ''
    model = _pipeline(model_name)
    results = []
    for chunk in chunk_text(text, max_words=220, overlap_words=25)[:max_chunks]:
        if model_name == 't5-small':
            chunk = 'summarize: ' + chunk
        result = model(chunk, max_length=120, min_length=15, do_sample=False, truncation=True)
        results.append(result[0]['summary_text'])
    return '\n\n'.join(results)


def word_count(text: str) -> int:
    return len(re.findall(r'\b\S+\b', text))
