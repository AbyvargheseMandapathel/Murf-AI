# app/services/llm.py

import json
import re
from types import SimpleNamespace
from typing import Iterator, List, Dict, Any, Optional

import requests
from app.config import settings

API_KEY = settings.GEMINI_API_KEY
API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-1.5-flash:streamGenerateContent?key={API_KEY}"
)

def _iter_gemini_events(resp: requests.Response) -> Iterator[Dict[str, Any]]:
    """
    Yield parsed JSON objects from Gemini stream.
    Supports both:
      1) SSE-style lines:  data: {...}\n\n ... data: [DONE]
      2) One-shot JSON array: [ {...}, {...} ]
    """
    buffer = []
    saw_data_lines = False

    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if not line:
            continue

        # SSE-style
        if line.startswith("data: "):
            saw_data_lines = True
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
                yield obj
            except json.JSONDecodeError:
                # Ignore malformed line
                continue
        else:
            # Non-SSE; collect to parse later as a whole JSON array
            buffer.append(line)

    # If no SSE used, try to parse the whole thing as a JSON array
    if not saw_data_lines and buffer:
        joined = "".join(buffer).strip()
        if joined:
            try:
                parsed = json.loads(joined)
                if isinstance(parsed, list):
                    for obj in parsed:
                        if isinstance(obj, dict):
                            yield obj
                elif isinstance(parsed, dict):
                    # Rare case: a single dict
                    yield parsed
            except json.JSONDecodeError:
                pass  # swallow if server sent something unexpected

def _parse_gemini_chunk(chunk: Dict[str, Any]) -> List[str]:
    """
    Extract all text parts from a response 'chunk' (dict).
    Looks into candidate.content.parts and candidate.delta.parts.
    """
    out: List[str] = []
    candidates = chunk.get("candidates") or []
    if not candidates:
        return out

    cand = candidates[0]

    # Full content parts
    for part in (cand.get("content") or {}).get("parts", []):
        if isinstance(part, dict) and "text" in part and isinstance(part["text"], str):
            out.append(part["text"])

    # Streaming delta parts (if any)
    for part in (cand.get("delta") or {}).get("parts", []):
        if isinstance(part, dict) and "text" in part and isinstance(part["text"], str):
            out.append(part["text"])

    return out

def query_gemini_stream(prompt: str) -> Iterator[SimpleNamespace]:
    """
    Calls Gemini and yields objects with a `.text` attribute (so your caller
    can safely do `for chunk in ...: print(chunk.text, end="")`).
    """
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 8192},
    }

    with requests.post(API_URL, json=payload, stream=True) as resp:
        resp.raise_for_status()
        for event in _iter_gemini_events(resp):
            for text_piece in _parse_gemini_chunk(event):
                # Yield an object with .text to match your consumer
                yield SimpleNamespace(text=text_piece)

def get_final_text(prompt: str) -> str:
    """Collects all streamed chunks into one final string."""
    return "".join(chunk.text for chunk in query_gemini_stream(prompt))

def extract_concise_answer(full_text: str) -> str:
    """
    Heuristically extract the short 'answer' from model text.
    - If markdown bold **Answer** is present, prefer that.
    - Else try common 'capital of X is Y' patterns.
    - Else return the trimmed full text.
    """
    if not full_text:
        return full_text

    # 1) Prefer markdown bold segment if it looks like the answer
    bold = re.findall(r"\*\*(.+?)\*\*", full_text)
    if bold:
        # pick the longest bold segment that is not a sentence
        bold_sorted = sorted(bold, key=len, reverse=True)
        for b in bold_sorted:
            if len(b.split()) <= 5:
                return b.strip()

    text = full_text.strip()

    # 2) Pattern: 'capital of <country> is <answer>'
    m = re.search(
        r"\bcapital of [A-Za-z\s\-]+ is ([A-Za-z\.\-\s]+?)(?:[.,;:\n]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    # 3) Pattern: '<answer> is the capital of <country>'
    m = re.search(
        r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\s+is the capital\b",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    # 4) Fallback: return the last sentence-ish fragment (trim markdown)
    cleaned = re.sub(r"[*_`]+", "", text).strip()
    sentences = re.split(r"[.!?\n]", cleaned)
    for s in reversed(sentences):
        s = s.strip()
        if s:
            return s
    return cleaned

# Example CLI run (optional)
if __name__ == "__main__":
    prompt = "What is the capital of India?"
    print("🚀 Streaming LLM response...\n" + "-" * 50)
    try:
        # Your existing consumer style will now work:
        for chunk in query_gemini_stream(prompt):
            print(chunk.text, end="", flush=True)

        final = get_final_text(prompt)
        concise = extract_concise_answer(final)
        print("\n" + "-" * 50)
        print("Full:", final.strip())
        print("Answer:", concise)

    except Exception as e:
        print(f"❌ LLM streaming error: {e}")