import re
from bisect import bisect_right
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

PAGE_MARKER_RE = re.compile(r'^\s*<!--\s*page:\s*(\d+)\s*-->\s*$')
EMPTY_PAGE_RE = re.compile(r'^\s*<!--\s*empty page\s*-->\s*$')
HEADING_RE = re.compile(r'^\s*#{1,6}\s*(.+?)\s*$')
FOOTNOTE_TITLE_RE = re.compile(r'^\(?[0-9٠-٩]+\)?$')
DIGITS_RE = re.compile(r'[0-9٠-٩]+')
TATWEEL_RE = re.compile(r'ـ+(?![ً-ٰٟۖ-ۭ])')

FOOTER_PATTERNS = [
    re.compile(r'^\s*-{3,}\s*$'),
    re.compile(r'^\s*[0-9٠-٩]+\s*$'),
    re.compile(r'^\s*<center>.*</center>\s*$'),
    re.compile(r'^.*مجمع فقهاء الشريعة بأمريكا.*$'),
    re.compile(r'^\s*(<span>)?\s*ما لا يسع المسلم جهله\s*(</span>)?\s*$'),
]

CHAPTER_PREFIX = 'الفصل'
TOC_TITLE = 'الفهرس'
EMPTY_HEADING_MIN_CHARS = 40


@dataclass
class ParsedParent:
    order: int
    title: str
    toc_title: str
    breadcrumb: List[str]
    lines: List[Tuple[int, str]] = field(default_factory=list)

    def build_text(self) -> Tuple[str, List[Tuple[int, int]]]:
        kept: List[str] = []
        page_offsets: List[Tuple[int, int]] = []
        offset = 0
        previous_blank = True
        for page, line in self.lines:
            line = line.rstrip()
            is_blank = not line.strip()
            if is_blank and previous_blank:
                continue
            if not is_blank and (not page_offsets or page_offsets[-1][1] != page):
                page_offsets.append((offset, page))
            kept.append(line)
            offset += len(line) + 1
            previous_blank = is_blank
        while kept and not kept[-1].strip():
            kept.pop()
        return '\n'.join(kept), page_offsets

    @property
    def text(self) -> str:
        return self.build_text()[0]

    @property
    def page_start(self) -> Optional[int]:
        pages = [p for p, line in self.lines if line.strip()]
        return pages[0] if pages else None

    @property
    def page_end(self) -> Optional[int]:
        pages = [p for p, line in self.lines if line.strip()]
        return pages[-1] if pages else None

    @property
    def metadata(self) -> dict:
        return {
            "title": self.title,
            "toc_title": self.toc_title,
            "breadcrumb": self.breadcrumb,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }


@dataclass
class BookParseResult:
    parents: List[ParsedParent]
    toc_titles: List[str]
    unmatched_toc_titles: List[str]
    dropped_duplicates: List[Tuple[int, str]]
    merged_into_previous: List[Tuple[int, str, str]]
    intro_parents: List[Tuple[int, str]]
    preamble_chars: int


def normalize(text: str) -> str:
    text = re.sub(r'[ً-ٰٟـ]', '', text)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = text.replace('ة', 'ه').replace('ى', 'ي')
    text = DIGITS_RE.sub('', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def strip_page_number(text: str) -> str:
    text = re.sub(r'^[0-9٠-٩]+\s+', '', text.strip())
    text = re.sub(r'\s+[0-9٠-٩]+$', '', text)
    return text.strip()


def is_footer(line: str) -> bool:
    return any(p.match(line) for p in FOOTER_PATTERNS)


def iter_page_lines(text: str):
    page = 0
    for line in text.split('\n'):
        marker = PAGE_MARKER_RE.match(line)
        if marker:
            page = int(marker.group(1))
            continue
        if EMPTY_PAGE_RE.match(line):
            continue
        yield page, TATWEEL_RE.sub('', line)


def extract_toc(lines, start_page: int, end_page: int):
    subsections, containers = [], []
    for page, line in lines:
        if not (start_page <= page <= end_page):
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith('|') or is_footer(stripped):
            continue
        heading = HEADING_RE.match(stripped)
        if heading:
            title = strip_page_number(heading.group(1))
            if normalize(title) and normalize(title) != normalize('الفهرس'):
                containers.append(title)
            continue
        title = strip_page_number(stripped)
        if len(normalize(title)) >= 3:
            subsections.append(title)
    return subsections, containers


def split_sections(lines, toc_start_page: int):
    preamble, sections, current = [], [], None
    for page, line in lines:
        if page >= toc_start_page:
            break
        stripped = line.strip()
        heading = HEADING_RE.match(stripped)
        if heading:
            title = heading.group(1).strip()
            if FOOTNOTE_TITLE_RE.match(title):
                continue
            current = {"title": title, "page": page, "lines": []}
            sections.append(current)
            continue
        if is_footer(stripped):
            continue
        (current["lines"] if current else preamble).append((page, line))
    return preamble, sections


def best_match(title: str, candidates_norm: List[str]) -> Tuple[Optional[int], float]:
    title_norm = normalize(title)
    best_idx, best_ratio = None, 0.0
    for idx, candidate in enumerate(candidates_norm):
        ratio = SequenceMatcher(None, title_norm, candidate).ratio()
        if ratio > best_ratio:
            best_idx, best_ratio = idx, ratio
    return best_idx, best_ratio


def page_at(page_offsets: List[Tuple[int, int]], position: int) -> Optional[int]:
    if not page_offsets:
        return None
    index = bisect_right([offset for offset, _ in page_offsets], position) - 1
    return page_offsets[max(index, 0)][1]


def find_toc_pages(lines) -> Tuple[Optional[int], Optional[int]]:
    toc_pages = []
    for page, line in lines:
        heading = HEADING_RE.match(line.strip())
        if heading and normalize(heading.group(1)).startswith(normalize(TOC_TITLE)):
            toc_pages.append(page)
    if not toc_pages:
        return None, None
    return min(toc_pages), max(toc_pages)


def content_size(section_lines) -> int:
    return len(''.join(line.strip() for _, line in section_lines))


def parse_markdown_book(text: str, toc_start_page: Optional[int] = None,
                        toc_end_page: Optional[int] = None,
                        threshold: float = 0.85) -> Optional[BookParseResult]:

    lines = list(iter_page_lines(text))
    if toc_start_page is None or toc_end_page is None:
        toc_start_page, toc_end_page = find_toc_pages(lines)
    if toc_start_page is None:
        return None
    toc_titles, toc_containers = extract_toc(lines, toc_start_page, toc_end_page)
    toc_norm = [normalize(t) for t in toc_titles]
    container_norm = [normalize(t) for t in toc_containers]

    preamble, sections = split_sections(lines, toc_start_page)

    for section in sections:
        idx, ratio = best_match(section["title"], toc_norm)
        section["toc_idx"] = idx if ratio >= threshold else None
        section["size"] = content_size(section["lines"])

    winner_for_toc = {}
    for i, section in enumerate(sections):
        idx = section["toc_idx"]
        if idx is None:
            continue
        current = winner_for_toc.get(idx)
        if current is None or section["size"] > sections[current]["size"]:
            winner_for_toc[idx] = i
    winners = set(winner_for_toc.values())

    for i, section in enumerate(sections):
        section["is_chapter"] = normalize(section["title"]).startswith(normalize(CHAPTER_PREFIX))
        _, container_ratio = best_match(section["title"], container_norm)
        section["is_container"] = (not section["is_chapter"]) and container_ratio >= threshold

    parents: List[ParsedParent] = []
    dropped_duplicates, merged_into_previous, intro_parents = [], [], []
    chapter, container = None, None
    preamble_chars = content_size(preamble)

    def add_parent(title, toc_title, page, section_lines):
        breadcrumb = [c for c in (chapter, container) if c and c != title]
        parents.append(ParsedParent(
            order=len(parents) + 1,
            title=title,
            toc_title=toc_title,
            breadcrumb=breadcrumb,
            lines=[(page, title)] + list(section_lines),
        ))

    for i, section in enumerate(sections):
        title = section["title"]

        if i in winners:
            add_parent(title, toc_titles[section["toc_idx"]], section["page"], section["lines"])
            continue

        if section["toc_idx"] is not None:
            dropped_duplicates.append((section["page"], title))

        if section["is_chapter"]:
            chapter, container = title, None
        elif section["is_container"]:
            container = title

        if section["size"] == 0:
            is_sentence = len(normalize(title)) >= EMPTY_HEADING_MIN_CHARS
            if (is_sentence and parents and section["toc_idx"] is None
                    and not section["is_chapter"] and not section["is_container"]):
                parents[-1].lines.append((section["page"], title))
                merged_into_previous.append((section["page"], title, parents[-1].title))
            continue

        if section["is_chapter"] or section["is_container"]:
            add_parent(title, title, section["page"], section["lines"])
            intro_parents.append((section["page"], title))
        elif parents:
            parents[-1].lines.append((section["page"], title))
            parents[-1].lines.extend(section["lines"])
            merged_into_previous.append((section["page"], title, parents[-1].title))
        else:
            preamble_chars += section["size"]

    covered = {sections[i]["toc_idx"] for i in winners}
    unmatched = [t for idx, t in enumerate(toc_titles) if idx not in covered]

    return BookParseResult(
        parents=parents,
        toc_titles=toc_titles,
        unmatched_toc_titles=unmatched,
        dropped_duplicates=dropped_duplicates,
        merged_into_previous=merged_into_previous,
        intro_parents=intro_parents,
        preamble_chars=preamble_chars,
    )
