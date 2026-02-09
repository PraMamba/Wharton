import re
from typing import List

from pypdf import PdfReader


def pdf_to_markdown(pdf_path: str) -> str:
    """Extract text from a PDF and convert it to Markdown."""
    text = _normalize_text(_extract_text(pdf_path))
    lines = [line.rstrip() for line in text.split("\n")]
    md_lines = _lines_to_markdown(lines)
    return _clean_markdown(md_lines)


def _extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        page_text = page.extract_text(extraction_mode="layout") or ""
        pages.append(page_text)
    return "\n\n".join(pages)


def _normalize_text(text: str) -> str:
    replacements = {
        "\u00a0": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
        "\u2022": "-",
        "\u25cf": "-",
        "¡ª": "--",
        "¡ñ": "-",
        "¨C": "-",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def _lines_to_markdown(lines: List[str]) -> List[str]:
    lines = _insert_bullets_for_unbulleted_lists(lines)
    md_lines: List[str] = []
    title_set = False
    subtitle_set = False
    paragraph_lines: List[str] = []
    list_lines: List[str] = []

    def flush_paragraph():
        nonlocal paragraph_lines
        if paragraph_lines:
            paragraph = _join_paragraph(paragraph_lines)
            if paragraph:
                md_lines.append(paragraph)
                md_lines.append("")
            paragraph_lines = []

    def flush_list():
        nonlocal list_lines
        if list_lines:
            md_lines.extend(_list_block_to_markdown(list_lines))
            md_lines.append("")
            list_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            flush_list()
            i += 1
            continue

        if _is_page_number_line(stripped):
            i += 1
            continue

        if not title_set:
            title = stripped
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                next_stripped = lines[j].strip()
                if title.endswith(":") and _looks_like_title_continuation(next_stripped):
                    title = f"{title} {next_stripped}"
                    i = j
            md_lines.append(f"# {title}")
            md_lines.append("")
            title_set = True
            i += 1
            continue

        if title_set and not subtitle_set and _looks_like_subtitle(stripped):
            md_lines.append(f"*{stripped}*")
            md_lines.append("")
            subtitle_set = True
            i += 1
            continue

        if _is_heading(stripped):
            flush_paragraph()
            flush_list()
            md_lines.append(f"{_heading_level(stripped)} {stripped}")
            md_lines.append("")
            i += 1
            continue

        if _is_list_line(line):
            flush_paragraph()
            list_lines.append(line)
            i += 1
            continue

        if list_lines:
            list_lines.append(line)
            i += 1
            continue

        paragraph_lines.append(stripped)
        i += 1

    flush_paragraph()
    flush_list()
    return md_lines


def _is_page_number_line(text: str) -> bool:
    return text.isdigit() and len(text) <= 3


def _is_heading(text: str) -> bool:
    if not text:
        return False
    if re.match(r"^(\d+\.|-|via\s)", text, re.IGNORECASE):
        return False
    if len(text) > 80:
        return False
    if text.endswith("."):
        return False
    if re.match(
        r"^(Phase\s+\d|Competition|Data Roadmap|Educational Modules|How to Succeed|About the World Hockey League|Main Competition Tasks|Predict|Create|Quantify|Communicate)",
        text,
    ):
        return True
    return False


def _heading_level(text: str) -> str:
    if text.startswith("Phase "):
        return "###"
    if text.startswith("Competition") or text.startswith("Data Roadmap") or text.startswith("Educational Modules"):
        return "##"
    if text.startswith("About the World Hockey League") or text.startswith("How to Succeed") or text.startswith(
        "Main Competition Tasks"
    ):
        return "##"
    return "###"


def _looks_like_subtitle(text: str) -> bool:
    return len(text) <= 40 and "workbook" in text.lower()


def _looks_like_title_continuation(text: str) -> bool:
    if not text or len(text) > 80:
        return False
    if re.match(r"^(\d+\.|-)", text):
        return False
    return True


def _is_list_line(line: str) -> bool:
    return bool(re.match(r"^\s*(\d+\.|-)\s+", line.strip()))


def _list_block_to_markdown(lines: List[str]) -> List[str]:
    md_lines: List[str] = []
    last_idx = None
    last_indent = 0
    for line in lines:
        if not line.strip():
            continue

        match = re.match(r"^(\s*)(\d+\.|-)\s+(.*)", line)
        if match:
            indent = len(match.group(1))
            marker = match.group(2)
            text = match.group(3).strip()
            indent_level = max(indent // 4, 0)
            prefix = ("  " * indent_level) + (marker if marker.endswith(".") else "-") + " "
            md_lines.append(prefix + text)
            last_idx = len(md_lines) - 1
            last_indent = indent
            continue

        stripped = line.strip()
        if last_idx is not None and (len(line) - len(line.lstrip(" "))) > last_indent:
            md_lines[last_idx] += " " + stripped
        else:
            md_lines.append(stripped)
            last_idx = len(md_lines) - 1
            last_indent = 0

    return md_lines


def _join_paragraph(lines: List[str]) -> str:
    parts = [line.strip() for line in lines if line.strip()]
    if not parts:
        return ""
    paragraph = " ".join(parts)
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    paragraph = _normalize_note_line(paragraph)
    paragraph = _normalize_submission_line(paragraph)
    return paragraph


def _normalize_note_line(paragraph: str) -> str:
    if paragraph.startswith("*Note"):
        paragraph = paragraph.lstrip("*").strip()
        if paragraph.lower().startswith("note"):
            paragraph = paragraph[4:].strip()
        if paragraph.lower().startswith("that "):
            paragraph = paragraph[5:].strip()
        return f"> Note: {paragraph}"
    return paragraph


def _normalize_submission_line(paragraph: str) -> str:
    if paragraph.startswith("Submission:"):
        return "**Submission:** " + paragraph[len("Submission:") :].strip()
    return paragraph


def _clean_markdown(md_lines: List[str]) -> str:
    cleaned = []
    for line in md_lines:
        cleaned.append(re.sub(r"\s+$", "", line))

    final_lines = []
    blank = False
    for line in cleaned:
        if line == "":
            if not blank:
                final_lines.append("")
            blank = True
        else:
            final_lines.append(line)
            blank = False
    return "\n".join(final_lines).strip() + "\n"


def _insert_bullets_for_unbulleted_lists(lines: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            out.append(line)
            i += 1
            continue

        if _is_heading(stripped) or _is_list_line(line):
            out.append(line)
            i += 1
            continue

        run = []
        j = i
        while j < len(lines):
            candidate = lines[j].strip()
            if not candidate:
                break
            if _is_heading(candidate) or _is_list_line(lines[j]):
                break
            if not _looks_like_list_candidate(candidate):
                break
            run.append(candidate)
            j += 1

        if len(run) >= 3:
            for item in run:
                out.append(f"- {item}")
            i = j
            continue

        out.append(line)
        i += 1

    return out


def _looks_like_list_candidate(text: str) -> bool:
    if len(text) > 70:
        return False
    if text.endswith("."):
        return False
    if text.endswith(":"):
        return False
    if not re.search(r"[A-Za-z]", text):
        return False
    return True
