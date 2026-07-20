from __future__ import annotations

import re


class MarkdownChunker:
    """Split markdown documents into overlapping chunks by heading boundaries."""

    def __init__(self, max_tokens: int = 512, overlap: int = 50):
        self.max_tokens = max_tokens
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        sections = self._split_by_headings(text)
        chunks: list[str] = []
        for section in sections:
            chunks.extend(self._split_section(section))
        return chunks

    def _split_by_headings(self, text: str) -> list[str]:
        pattern = re.compile(r"(?=^#{1,6}\s+)", re.MULTILINE)
        parts = pattern.split(text)
        return [p.strip() for p in parts if p.strip()]

    def _split_section(self, section: str) -> list[str]:
        words = section.split()
        if not words:
            return []
        if len(words) <= self.max_tokens:
            return [section]
        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + self.max_tokens, len(words))
            chunk = " ".join(words[start:end])
            if chunk:
                chunks.append(chunk)
            start += self.max_tokens - self.overlap
        return chunks
