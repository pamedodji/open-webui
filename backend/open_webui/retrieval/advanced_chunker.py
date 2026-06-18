"""Advanced chunker hook for OpenWebUI ingestion.

This module is loaded by a small OpenWebUI patch when
``OPENWEBUI_ADVANCED_CHUNKING=true`` is set.

It receives LangChain ``Document`` objects after OpenWebUI's loader extracted
text, and returns new ``Document`` objects ready for embedding. It intentionally
does not call OpenWebUI internals so it remains easy to test and revert.
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


DEFAULT_MAX_CHARS = 1400
DEFAULT_OVERLAP_CHARS = 180
DEFAULT_TABLE_ROWS = 1
DEFAULT_JSON_ITEMS = 8
DEFAULT_CSV_ROWS = 4
DEFAULT_JSONL_LINES = 5


def _get_config_int(request: Any, env_name: str, config_name: str, default: int) -> int:
    import os

    raw = os.getenv(env_name)
    if raw is None and request is not None:
        raw = getattr(getattr(request.app.state, "config", object()), config_name, None)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _source_name(metadata: dict[str, Any]) -> str:
    return str(
        metadata.get("name")
        or metadata.get("filename")
        or metadata.get("source")
        or metadata.get("title")
        or "document"
    )


def _source_suffix(metadata: dict[str, Any]) -> str:
    return Path(_source_name(metadata).split("?", 1)[0]).suffix.lower()


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    raw = text[4:end].strip()
    body = text[end + 5 :].lstrip()
    meta: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"')
    return meta, body


def _infer_title(text: str, metadata: dict[str, Any]) -> str:
    if metadata.get("title"):
        return str(metadata["title"])
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s+\S", line):
            return line.lstrip("#").strip()
    return Path(_source_name(metadata)).stem.replace("_", " ").replace("-", " ")


def _metadata_prefix(metadata: dict[str, Any], title: str, kind: str) -> str:
    source = _source_name(metadata)
    status = metadata.get("status") or metadata.get("source_status") or ""
    owner = metadata.get("owner") or metadata.get("source_owner") or ""
    headings = metadata.get("headings")
    if isinstance(headings, list):
        headings_text = " > ".join(str(h) for h in headings)
    else:
        headings_text = str(headings or "")

    parts = [
        f"chunk_title: {title}",
        f"source_name: {source}",
        f"source_status: {status}",
        f"source_owner: {owner}",
        f"chunk_kind: {kind}",
    ]
    if headings_text:
        parts.append(f"source_headings: {headings_text}")

    return "---\n" + "\n".join(parts) + "\n---\n\n" + f"# {title}\n\nSource: `{source}`\n\n"


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and "|" in stripped[1:-1]


def _is_table_separator(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def _split_table_blocks(lines: list[str]) -> tuple[list[list[str]], list[str]]:
    tables: list[list[str]] = []
    non_table: list[str] = []
    idx = 0
    while idx < len(lines):
        if idx + 1 < len(lines) and _is_table_row(lines[idx]) and _is_table_separator(lines[idx + 1]):
            block = [lines[idx], lines[idx + 1]]
            idx += 2
            while idx < len(lines) and _is_table_row(lines[idx]):
                block.append(lines[idx])
                idx += 1
            tables.append(block)
            non_table.append("")
            continue
        non_table.append(lines[idx])
        idx += 1
    return tables, non_table


def _table_cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _row_facts(headers: list[str], row: list[str]) -> str:
    facts = []
    for key, value in zip(headers, row):
        if key and value:
            facts.append(f"- {key}: {value}")
    return "\n".join(facts)


def _compact_scalar(value: Any, max_len: int = 240) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value).replace("\n", " ").strip()
    if len(text) > max_len:
        return text[: max_len - 3].rstrip() + "..."
    return text


def _flatten_json(value: Any, prefix: str = "", limit: int = 80) -> list[tuple[str, str]]:
    facts: list[tuple[str, str]] = []

    def walk(item: Any, path: str) -> None:
        if len(facts) >= limit:
            return
        if isinstance(item, dict):
            for key, child in item.items():
                child_path = f"{path}.{key}" if path else str(key)
                walk(child, child_path)
        elif isinstance(item, list):
            if not item:
                facts.append((path, "[]"))
                return
            for idx, child in enumerate(item[:10]):
                walk(child, f"{path}[{idx}]")
        else:
            facts.append((path or "value", _compact_scalar(item)))

    walk(value, prefix)
    return facts


def _json_facts(value: Any, prefix: str = "") -> str:
    return "\n".join(f"- {path}: {val}" for path, val in _flatten_json(value, prefix=prefix))


def _looks_like_csv(text: str) -> bool:
    sample = "\n".join(line for line in text.splitlines()[:10] if line.strip())
    if not sample or "," not in sample:
        return False
    try:
        dialect = csv.Sniffer().sniff(sample)
        rows = list(csv.reader(io.StringIO(sample), dialect))
    except csv.Error:
        return False
    return len(rows) >= 2 and len(rows[0]) > 1


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = "Overview"
    current_lines: list[str] = []
    for line in text.splitlines():
        # Avoid treating shell comments or #SBATCH lines as Markdown headings.
        if re.match(r"^#{1,6}\s+\S", line):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line.lstrip("#").strip() or "Section"
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))
    return [(title, "\n".join(lines).strip()) for title, lines in sections if "\n".join(lines).strip()]


def _split_by_size(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            newline = text.rfind("\n", start, end)
            if newline > start + max_chars // 2:
                end = newline
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)
    return [chunk for chunk in chunks if chunk]


def _copy_metadata(metadata: dict[str, Any], strategy: str, index: int, title: str) -> dict[str, Any]:
    copied = dict(metadata)
    copied["advanced_chunking"] = True
    copied["advanced_chunk_strategy"] = strategy
    copied["advanced_chunk_index"] = index
    copied["advanced_chunk_title"] = title
    return copied


def _append_chunk(
    output: list[Document],
    metadata: dict[str, Any],
    strategy: str,
    title: str,
    content: str,
    index: int,
) -> None:
    output.append(
        Document(
            page_content=content,
            metadata=_copy_metadata(metadata, strategy, index, title),
        )
    )


def _chunk_markdown_text(
    text: str,
    metadata: dict[str, Any],
    max_chars: int,
    overlap_chars: int,
    table_rows: int,
    start_index: int,
) -> tuple[list[Document], int]:
    source_title = _infer_title(text, metadata)
    tables, non_table_lines = _split_table_blocks(text.splitlines())
    non_table_text = "\n".join(non_table_lines).strip()
    chunks: list[Document] = []
    chunk_index = start_index

    for title, section in _split_markdown_sections(non_table_text):
        for part in _split_by_size(section, max_chars=max_chars, overlap_chars=overlap_chars):
            chunk_index += 1
            chunk_title = source_title if title == "Overview" else (title or source_title)
            content = _metadata_prefix(metadata, chunk_title, "markdown-section") + part
            _append_chunk(chunks, metadata, "markdown-section", chunk_title, content, chunk_index)

    for block in tables:
        if len(block) < 3:
            continue
        header_line, separator_line, *rows = block
        headers = _table_cells(header_line)
        for start in range(0, len(rows), table_rows):
            batch = rows[start : start + table_rows]
            first_cells = _table_cells(batch[0]) if batch else []
            row_label = next((cell for cell in first_cells if cell), f"rows-{start + 1}")
            chunk_title = f"{source_title} - table {row_label}"
            facts = "\n\n".join(_row_facts(headers, _table_cells(row)) for row in batch)
            original = "\n".join([header_line, separator_line, *batch])
            content = (
                _metadata_prefix(metadata, chunk_title, "table-row")
                + "Structured row facts:\n"
                + facts
                + "\n\nOriginal table row(s):\n\n"
                + original
            )
            chunk_index += 1
            _append_chunk(chunks, metadata, "table-row", chunk_title, content, chunk_index)

    return chunks, chunk_index


def _chunk_json_text(
    text: str,
    metadata: dict[str, Any],
    max_chars: int,
    overlap_chars: int,
    max_items: int,
    start_index: int,
) -> tuple[list[Document], int]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [], start_index

    chunks: list[Document] = []
    chunk_index = start_index
    source_title = _infer_title(text, metadata)

    if isinstance(parsed, dict):
        items = list(parsed.items())
        for key, value in items:
            rendered = json.dumps({key: value}, ensure_ascii=False, indent=2, sort_keys=True)
            for part_idx, part in enumerate(_split_by_size(rendered, max_chars, overlap_chars), 1):
                chunk_index += 1
                chunk_title = f"{source_title} - {key}" if part_idx == 1 else f"{source_title} - {key} part {part_idx}"
                content = (
                    _metadata_prefix(metadata, chunk_title, "json-object")
                    + "Structured JSON facts:\n"
                    + _json_facts(value, prefix=str(key))
                    + "\n\nOriginal JSON fragment:\n\n```json\n"
                    + part
                    + "\n```"
                )
                _append_chunk(chunks, metadata, "json-object", chunk_title, content, chunk_index)
    elif isinstance(parsed, list):
        for start in range(0, len(parsed), max_items):
            batch = parsed[start : start + max_items]
            rendered = json.dumps(batch, ensure_ascii=False, indent=2, sort_keys=True)
            chunk_index += 1
            chunk_title = f"{source_title} items {start + 1}-{start + len(batch)}"
            facts = "\n\n".join(_json_facts(item, prefix=f"item[{start + idx}]") for idx, item in enumerate(batch))
            content = (
                _metadata_prefix(metadata, chunk_title, "json-array")
                + "Structured JSON facts:\n"
                + facts
                + "\n\nOriginal JSON fragment:\n\n```json\n"
                + rendered
                + "\n```"
            )
            _append_chunk(chunks, metadata, "json-array", chunk_title, content, chunk_index)
    else:
        rendered = json.dumps(parsed, ensure_ascii=False, indent=2)
        chunk_index += 1
        chunk_title = source_title
        content = _metadata_prefix(metadata, chunk_title, "json-value") + f"JSON value: {_compact_scalar(parsed)}\n\n```json\n{rendered}\n```"
        _append_chunk(chunks, metadata, "json-value", chunk_title, content, chunk_index)

    return chunks, chunk_index


def _chunk_jsonl_text(text: str, metadata: dict[str, Any], max_lines: int, start_index: int) -> tuple[list[Document], int]:
    lines = [line for line in text.splitlines() if line.strip()]
    chunks: list[Document] = []
    chunk_index = start_index
    source_title = _infer_title(text, metadata)

    for start in range(0, len(lines), max_lines):
        batch = lines[start : start + max_lines]
        facts = []
        for offset, line in enumerate(batch):
            try:
                facts.append(_json_facts(json.loads(line), prefix=f"line[{start + offset + 1}]"))
            except json.JSONDecodeError:
                facts.append(f"- line[{start + offset + 1}]: {_compact_scalar(line)}")
        chunk_index += 1
        chunk_title = f"{source_title} lines {start + 1}-{start + len(batch)}"
        content = (
            _metadata_prefix(metadata, chunk_title, "jsonl-batch")
            + "Structured JSONL facts:\n"
            + "\n\n".join(facts)
        )
        _append_chunk(chunks, metadata, "jsonl-batch", chunk_title, content, chunk_index)

    return chunks, chunk_index


def _chunk_csv_text(
    text: str,
    metadata: dict[str, Any],
    max_rows: int,
    start_index: int,
) -> tuple[list[Document], int]:
    try:
        sample = "\n".join(line for line in text.splitlines()[:20] if line.strip())
        dialect = csv.Sniffer().sniff(sample) if sample else csv.excel
        reader = csv.DictReader(io.StringIO(text), dialect=dialect)
        rows = list(reader)
    except csv.Error:
        return [], start_index
    if not rows or not reader.fieldnames:
        return [], start_index

    chunks: list[Document] = []
    chunk_index = start_index
    source_title = _infer_title(text, metadata)
    headers = [header or f"column_{idx + 1}" for idx, header in enumerate(reader.fieldnames)]

    for start in range(0, len(rows), max_rows):
        batch = rows[start : start + max_rows]
        row_label = next((_compact_scalar(value, 60) for value in batch[0].values() if value), f"rows {start + 1}")
        facts = []
        for idx, row in enumerate(batch, start + 1):
            facts.append(f"row {idx}:")
            facts.extend(f"- {key}: {_compact_scalar(row.get(key, ''))}" for key in headers if row.get(key, ""))
        original = io.StringIO()
        writer = csv.DictWriter(original, fieldnames=headers)
        writer.writeheader()
        writer.writerows(batch)
        chunk_index += 1
        chunk_title = f"{source_title} - row {row_label}"
        content = (
            _metadata_prefix(metadata, chunk_title, "csv-row")
            + "Structured CSV facts:\n"
            + "\n".join(facts)
            + "\n\nOriginal CSV row(s):\n\n```csv\n"
            + original.getvalue().strip()
            + "\n```"
        )
        _append_chunk(chunks, metadata, "csv-row", chunk_title, content, chunk_index)

    return chunks, chunk_index


def _chunk_yaml_text(text: str, metadata: dict[str, Any], max_chars: int, overlap_chars: int, start_index: int) -> tuple[list[Document], int]:
    lines = text.splitlines()
    blocks: list[tuple[str, list[str]]] = []
    current_title = "Overview"
    current: list[str] = []

    for line in lines:
        if re.match(r"^[A-Za-z0-9_.-][^:#]*:\s*", line):
            if current:
                blocks.append((current_title, current))
            current_title = line.split(":", 1)[0].strip() or "Section"
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append((current_title, current))

    chunks: list[Document] = []
    chunk_index = start_index
    source_title = _infer_title(text, metadata)
    for title, block_lines in blocks:
        block = "\n".join(block_lines).strip()
        for part_idx, part in enumerate(_split_by_size(block, max_chars, overlap_chars), 1):
            chunk_index += 1
            chunk_title = f"{source_title} - {title}" if title != "Overview" else source_title
            if part_idx > 1:
                chunk_title = f"{chunk_title} part {part_idx}"
            facts = []
            for raw in part.splitlines():
                line = raw.strip().lstrip("- ").strip()
                if ":" in line:
                    key, value = line.split(":", 1)
                    if value.strip():
                        facts.append(f"- {key.strip()}: {value.strip()}")
            content = (
                _metadata_prefix(metadata, chunk_title, "yaml-block")
                + ("Structured YAML facts:\n" + "\n".join(facts) + "\n\n" if facts else "")
                + "Original YAML block:\n\n```yaml\n"
                + part
                + "\n```"
            )
            _append_chunk(chunks, metadata, "yaml-block", chunk_title, content, chunk_index)

    return chunks, chunk_index


def _split_pdf_pages(text: str) -> list[str]:
    if "\f" in text:
        return [page.strip() for page in text.split("\f") if page.strip()]
    page_pattern = re.compile(r"(?=^\s*(?:Page|PAGE)\s+\d+(?:\s+of\s+\d+)?\s*$)", re.MULTILINE)
    pages = [page.strip() for page in page_pattern.split(text) if page.strip()]
    return pages if len(pages) > 1 else [text.strip()]


def _chunk_pdf_text(
    text: str,
    metadata: dict[str, Any],
    max_chars: int,
    overlap_chars: int,
    table_rows: int,
    start_index: int,
) -> tuple[list[Document], int]:
    chunks: list[Document] = []
    chunk_index = start_index
    source_title = _infer_title(text, metadata)

    for page_idx, page in enumerate(_split_pdf_pages(text), 1):
        page_metadata = {**metadata, "pdf_page": page_idx}
        md_chunks, chunk_index = _chunk_markdown_text(page, page_metadata, max_chars, overlap_chars, table_rows, chunk_index)
        if md_chunks:
            for chunk in md_chunks:
                chunk.metadata["advanced_chunk_strategy"] = "pdf-" + str(chunk.metadata.get("advanced_chunk_strategy", "text"))
            chunks.extend(md_chunks)
            continue
        for part_idx, part in enumerate(_split_by_size(page, max_chars, overlap_chars), 1):
            chunk_index += 1
            chunk_title = f"{source_title} page {page_idx}" if part_idx == 1 else f"{source_title} page {page_idx} part {part_idx}"
            content = _metadata_prefix(page_metadata, chunk_title, "pdf-page") + part
            _append_chunk(chunks, page_metadata, "pdf-page", chunk_title, content, chunk_index)

    return chunks, chunk_index


def split_documents(docs: list[Document], request: Any = None) -> list[Document]:
    max_chars = _get_config_int(
        request,
        "OPENWEBUI_ADVANCED_CHUNK_MAX_CHARS",
        "CHUNK_SIZE",
        DEFAULT_MAX_CHARS,
    )
    overlap_chars = _get_config_int(
        request,
        "OPENWEBUI_ADVANCED_CHUNK_OVERLAP_CHARS",
        "CHUNK_OVERLAP",
        DEFAULT_OVERLAP_CHARS,
    )
    table_rows = _get_config_int(
        request,
        "OPENWEBUI_ADVANCED_CHUNK_TABLE_ROWS",
        "ADVANCED_CHUNK_TABLE_ROWS",
        DEFAULT_TABLE_ROWS,
    )
    table_rows = max(1, table_rows)

    output: list[Document] = []
    chunk_index = 0

    for doc in docs:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        text = str(getattr(doc, "page_content", "") or "").strip()
        if not text:
            continue

        frontmatter, text = _parse_frontmatter(text)
        metadata = {**frontmatter, **metadata}

        source_title = _infer_title(text, metadata)
        tables, non_table_lines = _split_table_blocks(text.splitlines())
        non_table_text = "\n".join(non_table_lines).strip()

        for title, section in _split_markdown_sections(non_table_text):
            for part in _split_by_size(section, max_chars=max_chars, overlap_chars=overlap_chars):
                chunk_index += 1
                chunk_title = title or source_title
                content = _metadata_prefix(metadata, chunk_title, "markdown-section") + part
                output.append(
                    Document(
                        page_content=content,
                        metadata=_copy_metadata(metadata, "markdown-section", chunk_index, chunk_title),
                    )
                )

        for block in tables:
            if len(block) < 3:
                continue
            header_line, separator_line, *rows = block
            headers = _table_cells(header_line)
            for start in range(0, len(rows), table_rows):
                batch = rows[start : start + table_rows]
                first_cells = _table_cells(batch[0]) if batch else []
                row_label = next((cell for cell in first_cells if cell), f"rows-{start + 1}")
                chunk_title = f"{source_title} - table {row_label}"
                facts = "\n\n".join(_row_facts(headers, _table_cells(row)) for row in batch)
                original = "\n".join([header_line, separator_line, *batch])
                content = (
                    _metadata_prefix(metadata, chunk_title, "table-row")
                    + "Structured row facts:\n"
                    + facts
                    + "\n\nOriginal table row(s):\n\n"
                    + original
                )
                chunk_index += 1
                output.append(
                    Document(
                        page_content=content,
                        metadata=_copy_metadata(metadata, "table-row", chunk_index, chunk_title),
                    )
                )

        if not output:
            for part in _split_by_size(text, max_chars=max_chars, overlap_chars=overlap_chars):
                chunk_index += 1
                content = _metadata_prefix(metadata, source_title, "plain-text") + part
                output.append(
                    Document(
                        page_content=content,
                        metadata=_copy_metadata(metadata, "plain-text", chunk_index, source_title),
                    )
                )

    return output
