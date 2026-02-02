import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class Chunker:
    def chunk(self, content: str, file_path: str) -> List[Dict]:
        raise NotImplementedError


class MarkdownChunker(Chunker):
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, content: str, file_path: str) -> List[Dict]:
        chunks = []
        sections = re.split(r'\n(?=#{1,3}\s+[^\n]+)', content)

        for i, section in enumerate(sections):
            if not section.strip():
                continue

            if len(section) > self.chunk_size * 2:
                paragraphs = re.split(r'\n\n+', section)
                current_chunk = ""

                for para in paragraphs:
                    if len(current_chunk) + len(para) >= self.chunk_size:
                        if current_chunk:
                            chunks.append({
                                "content": current_chunk.strip(),
                                "source_file": file_path,
                                "metadata": {
                                    "chunk_type": "markdown",
                                    "section_index": i,
                                }
                            })
                        if self.overlap > 0 and current_chunk:
                            current_chunk = current_chunk[-self.overlap:] + "\n\n"
                        else:
                            current_chunk = ""
                    current_chunk += para + "\n\n"

                if current_chunk.strip():
                    chunks.append({
                        "content": current_chunk.strip(),
                        "source_file": file_path,
                        "metadata": {
                            "chunk_type": "markdown",
                            "section_index": i,
                        }
                    })
            else:
                chunks.append({
                    "content": section.strip(),
                    "source_file": file_path,
                    "metadata": {
                        "chunk_type": "markdown",
                        "section_index": i,
                    }
                })

        return chunks


class CodeChunker(Chunker):
    PATTERNS: Dict[str, List[str]] = {
        "python": [
            r'\n\s*(def|class)\s+\w+',
            r'\n\s*(def|class)\s+\w+\s*\(.*\):',
        ],
        "java": [
            r'\n\s*(public|private|protected)?\s*(static)?\s*(class|interface|enum)\s+\w+',
            r'\n\s*@.*\n\s*(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(.*\)',
        ],
        "c_cpp": [
            r'\n\s*(struct|class|interface|enum)\s+\w+',
            r'\n\s*(extern\s+)?"[C]"?\s*#include',
        ],
        "javascript": [
            r'\n\s*(function|const|let|var)\s+\w+\s*=\s*(async\s*)?\(',
            r'\n\s*(export\s+)?(default\s+)?class\s+\w+',
            r'\n\s*(export\s+)?(async\s+)?function\s+\w+\s*\(',
        ],
        "go": [
            r'\n\s*(func|type|interface)\s+\w+',
        ],
        "rust": [
            r'\n\s*(fn|struct|enum|trait|impl)\s+\w+',
        ],
    }

    GENERIC_PATTERNS = [
        r'\n\s*[a-zA-Z_]\w*\s*[a-zA-Z_]\w*\s*\([^)]*\)[^{]*\{',
        r'\n\s*\}\s*\n',
        r'\n\s*(public|private|protected|static|final|abstract)\s+',
    ]

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def detect_language(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        lang_map = {
            '.py': 'python',
            '.java': 'java',
            '.c': 'c_cpp',
            '.cpp': 'c_cpp',
            '.cc': 'c_cpp',
            '.h': 'c_cpp',
            '.hpp': 'c_cpp',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'javascript',
            '.tsx': 'javascript',
            '.go': 'go',
            '.rs': 'rust',
            '.cs': 'c_cpp',
            '.swift': 'c_cpp',
            '.kt': 'java',
        }
        return lang_map.get(ext, 'generic')

    def get_split_patterns(self, language: str) -> List[str]:
        if language in self.PATTERNS and language != 'generic':
            return self.PATTERNS[language]
        return self.GENERIC_PATTERNS

    def chunk(self, content: str, file_path: str) -> List[Dict]:
        language = self.detect_language(file_path)
        patterns = self.get_split_patterns(language)

        chunks = []
        split_positions = []

        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                split_positions.append(match.start())

        split_positions = sorted(set(split_positions))

        if not split_positions:
            return self._newline_fallback(content, file_path, language)

        prev_pos = 0
        for pos in split_positions:
            chunk_text = content[prev_pos:pos].strip()

            if len(chunk_text) > self.chunk_size or prev_pos == 0:
                if len(chunk_text) > self.chunk_size:
                    sub_chunks = self._split_large_chunk(
                        chunk_text, file_path, language
                    )
                    chunks.extend(sub_chunks)
                else:
                    if chunk_text:
                        chunks.append({
                            "content": chunk_text,
                            "source_file": file_path,
                            "metadata": {
                                "chunk_type": "code",
                                "language": language,
                            }
                        })
                prev_pos = pos

        remaining = content[prev_pos:].strip()
        if remaining:
            chunks.append({
                "content": remaining,
                "source_file": file_path,
                "metadata": {
                    "chunk_type": "code",
                    "language": language,
                }
            })

        return chunks

    def _newline_fallback(self, content: str, file_path: str, language: str) -> List[Dict]:
        chunks = []
        logical_blocks = re.split(r'\n\s*\n+', content)

        current_chunk = ""
        for block in logical_blocks:
            block = block.strip()
            if not block:
                continue

            if len(current_chunk) + len(block) >= self.chunk_size:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk,
                        "source_file": file_path,
                        "metadata": {
                            "chunk_type": "code",
                            "language": language,
                        }
                    })
                if self.overlap > 0 and current_chunk:
                    current_chunk = current_chunk[-self.overlap:] + "\n\n"
                else:
                    current_chunk = ""
            current_chunk += block + "\n\n"

        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "source_file": file_path,
                "metadata": {
                    "chunk_type": "code",
                    "language": language,
                }
            })

        return chunks

    def _split_large_chunk(self, text: str, file_path: str, language: str) -> List[Dict]:
        chunks = []
        lines = text.split('\n')
        current_chunk = ""

        for line in lines:
            if len(current_chunk) + len(line) >= self.chunk_size:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk,
                        "source_file": file_path,
                        "metadata": {
                            "chunk_type": "code",
                            "language": language,
                        }
                    })
                if self.overlap > 0 and current_chunk:
                    chunk_lines = current_chunk.split('\n')
                    overlap_lines = []
                    total_len = 0
                    for l in reversed(chunk_lines):
                        if total_len + len(l) >= self.overlap:
                            break
                        overlap_lines.insert(0, l)
                        total_len += len(l)
                    current_chunk = "\n".join(overlap_lines) + "\n"
                else:
                    current_chunk = ""
            current_chunk += line + "\n"

        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "source_file": file_path,
                "metadata": {
                    "chunk_type": "code",
                    "language": language,
                }
            })

        return chunks


class TextChunker(Chunker):
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, content: str, file_path: str) -> List[Dict]:
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', content)

        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) >= self.chunk_size:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "source_file": file_path,
                        "metadata": {
                            "chunk_type": "text",
                        }
                    })
                if self.overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.overlap:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "source_file": file_path,
                "metadata": {
                    "chunk_type": "text",
                }
            })

        return chunks


def get_chunker(file_path: str, chunk_size: int = 1000, overlap: int = 100) -> Chunker:
    ext = Path(file_path).suffix.lower()

    if ext in ['.md', '.mdx', '.markdown']:
        return MarkdownChunker(chunk_size, overlap)
    elif ext in ['.txt', '.csv', '.log', '.conf', '.ini', '.cfg', '.yaml', '.yml', '.json', '.toml']:
        return TextChunker(chunk_size, overlap)
    else:
        return CodeChunker(chunk_size, overlap)