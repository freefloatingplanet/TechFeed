import html
import os
from typing import Iterable, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

GT_ENDPOINT = "https://translation.googleapis.com/language/translate/v2"
DEFAULT_MAX_CHARS = 4500


def split_chunks(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> List[str]:
    """Translate APIで扱えるサイズに文字列を分割する。"""
    text = text or ""
    if not text:
        return []

    parts: List[str] = []
    buf: List[str] = []
    current = 0
    for line in text.splitlines(True):
        if current + len(line) > max_chars and buf:
            parts.append("".join(buf))
            buf = [line]
            current = len(line)
        else:
            buf.append(line)
            current += len(line)
    if buf:
        parts.append("".join(buf))
    return parts


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=20))
def _translate_chunk(text: str, *, target: str, source: Optional[str], api_key: str) -> str:
    params = {"key": api_key}
    data = {
        "q": text,
        "target": target,
        "format": "text",
    }
    if source:
        data["source"] = source

    response = requests.post(GT_ENDPOINT, params=params, data=data, timeout=60)
    if response.status_code >= 400:
        raise requests.HTTPError(
            f"Google Translate API error: {response.status_code} {response.text[:200]}"
        )
    translated = response.json()["data"]["translations"][0]["translatedText"]
    return html.unescape(translated)


def _get_api_key(explicit: Optional[str] = None) -> str:
    api_key = explicit or os.environ.get("GOOGLE_TRANSLATE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_TRANSLATE_API_KEY が設定されていません。翻訳機能を利用するには環境変数をセットしてください。"
        )
    return api_key


def _translate_text_with_key(text: str, target: str, source: Optional[str], api_key: str) -> str:
    chunks = split_chunks(text)
    if not chunks:
        return ""
    translated_parts = [
        _translate_chunk(chunk, target=target, source=source, api_key=api_key)
        for chunk in chunks
    ]
    return "".join(translated_parts)


def translate_text(
    text: str,
    *,
    target: str = "ja",
    source: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """単一テキストを翻訳して返す。"""
    text = text or ""
    if not text:
        return ""

    key = _get_api_key(api_key)
    return _translate_text_with_key(text, target, source, key)


def translate_texts(
    texts: Iterable[str],
    *,
    target: str = "ja",
    source: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[str]:
    """複数テキストを順に翻訳してリストで返す。"""
    results: List[str] = []
    key: Optional[str] = None

    for text in texts or []:
        if not text:
            results.append(text or "")
            continue
        if key is None:
            key = _get_api_key(api_key)
        results.append(_translate_text_with_key(text, target, source, key))
    return results
