#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Donor_Merger_GPT.py - Überarbeitet (Robustere Block-Erkennung + Matching)
Leg es neben: donor/ original/ backup  (oder benutze --donor/--orig/--backup)
"""
from __future__ import annotations
import os
import sys
import re
import shutil
import difflib
import argparse
import datetime
import logging
from typing import List, Tuple, Optional

# ----------------- Basis / Pfad -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_dir_prefer_local(path: Optional[str]) -> str:
    if path:
        cand = os.path.abspath(path)
        if os.path.isdir(cand):
            return cand
        rel = os.path.join(BASE_DIR, path)
        if os.path.isdir(rel):
            return os.path.abspath(rel)
        return cand
    return ""

def auto_detect_layout(donor_arg: Optional[str], orig_arg: Optional[str], backup_arg: Optional[str]):
    # priority: explicit args > local ./donor > ./ZUTUN/donor
    if donor_arg or orig_arg or backup_arg:
        d = resolve_dir_prefer_local(donor_arg or "donor")
        o = resolve_dir_prefer_local(orig_arg or "original")
        b = resolve_dir_prefer_local(backup_arg or "backup")
        return d, o, b

    local_d = os.path.join(BASE_DIR, "donor")
    local_o = os.path.join(BASE_DIR, "original")
    local_b = os.path.join(BASE_DIR, "backup")
    if os.path.isdir(local_d) and os.path.isdir(local_o):
        return local_d, local_o, local_b

    zutun_d = os.path.join(BASE_DIR, "ZUTUN", "donor")
    zutun_o = os.path.join(BASE_DIR, "ZUTUN", "original")
    zutun_b = os.path.join(BASE_DIR, "ZUTUN", "backup")
    if os.path.isdir(zutun_d) and os.path.isdir(zutun_o):
        return zutun_d, zutun_o, zutun_b

    # fallback to local candidates (may not exist)
    return local_d, local_o, local_b

# ----------------- Logging -----------------
LOGFILE = os.path.join(BASE_DIR, "merger_debug.log")
os.makedirs(os.path.dirname(LOGFILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOGFILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DonorMerger")

# ----------------- Config -----------------
CONTEXT_LINES = 8           # mehr Kontext sammeln hilft bei fragilen Donor-Blöcken
BACKWARD_CODE_LINES = 8    # falls kein code after marker: take up to N lines before marker
FUZZY_THRESHOLD = 0.55     # etwas großzügiger
ENCODINGS_TRY = ["utf-8", "cp949", "euc-kr", "latin-1", "cp1252"]
SUPPORTED_EXTS = (".cpp", ".c", ".h", ".hpp", ".txt")

# ----------------- IO Helpers -----------------
def try_read_file(path: str) -> Tuple[Optional[str], Optional[str]]:
    for enc in ENCODINGS_TRY:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read(), enc
        except Exception:
            continue
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), "utf-8-replace"
    except Exception as e:
        logger.exception(f"Fehler beim Lesen {path}: {e}")
        return None, None

def write_file_with_encoding(path: str, text: str, encoding: str = "utf-8"):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding=encoding, errors="replace") as f:
        f.write(text)
    os.replace(tmp, path)

def make_backup(orig_path: str, backup_dir: str) -> Optional[str]:
    try:
        os.makedirs(backup_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.basename(orig_path)
        dest = os.path.join(backup_dir, f"{base}.bak.{ts}")
        shutil.copy2(orig_path, dest)
        logger.info(f"Backup erstellt: {dest}")
        return dest
    except Exception as e:
        logger.exception(f"Backup fehlgeschlagen für {orig_path}: {e}")
        return None

# ----------------- Normalization & Regex -----------------
def strip_inline_comments(line: str) -> str:
    s = re.sub(r'//.*$', '', line)
    s = re.sub(r'/\*.*?\*/', '', s)
    return s

def normalize_for_search(line: str) -> str:
    s = strip_inline_comments(line)
    s = s.replace('{', ' ').replace('}', ' ').replace('(', ' ').replace(')', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()

def build_flexible_regex_from_line(line: str) -> str:
    nl = normalize_for_search(line)
    tokens = [re.escape(t) for t in nl.split()]
    if not tokens:
        return r'.*'
    pattern = r'\b' + r'\s+'.join(tokens) + r'\b'
    pattern = pattern.rstrip(r'\b') + r'(?:\s*[:{]\s*)?'
    return pattern

# ----------------- Indentation & Preprocessor -----------------
def preserve_indentation(reference_line: Optional[str], code_block: List[str]) -> List[str]:
    ref_indent = ''
    if reference_line is not None:
        m = re.match(r'^(\s*)', reference_line)
        if m:
            ref_indent = m.group(1)
    indent_unit = '\t' if '\t' in ref_indent else ' ' * 4
    increase = 0
    if reference_line and reference_line.rstrip().endswith('{'):
        increase = 1
    base_indent = ref_indent + (indent_unit * increase)
    indented = []
    for line in code_block:
        if line.strip() == '':
            indented.append(line)
            continue
        stripped = line.lstrip()
        if stripped.startswith('#'):
            indented.append(stripped)
        else:
            indented.append(base_indent + stripped)
    return indented

def adjust_for_preprocessor(original_lines: List[str], insert_idx: int) -> int:
    stack = []
    for i in range(0, min(insert_idx + 1, len(original_lines))):
        l = original_lines[i].strip()
        if re.match(r'#\s*ifn?def\b|#\s*if\b|#\s*ifdef\b', l):
            stack.append((i, l))
        elif re.match(r'#\s*endif\b', l):
            if stack:
                stack.pop()
    if stack:
        last_open_idx = stack[-1][0]
        return max(insert_idx, last_open_idx + 1)
    return insert_idx

# ----------------- Matching Algorithms (improved) -----------------
def find_insert_position_improved(original_lines: List[str], context_before: List[str]) -> int:
    if not context_before:
        return -1
    search_context = [l for l in context_before if l.strip()]
    search_context = search_context[-6:]
    if not search_context:
        return -1
    norm_orig = [normalize_for_search(l) for l in original_lines]
    ctx_norm = [normalize_for_search(l) for l in search_context]

    best_pos = -1
    best_score = 0

    # For each line in original, compute match score using nearby window
    N = len(original_lines)
    for i in range(N):
        score = 0
        # check up to len(ctx_norm) preceding lines aligning with search_context
        for offset, ctx in enumerate(reversed(ctx_norm)):
            check_idx = i - offset
            if check_idx < 0:
                break
            pat = build_flexible_regex_from_line(ctx)
            if re.search(pat, original_lines[check_idx], flags=re.IGNORECASE):
                score += 3
            else:
                # token-based partial match
                if ctx and ctx in norm_orig[check_idx]:
                    score += 1
                else:
                    # try searching in a local window +/-2 lines
                    window_hits = 0
                    for w in range(max(0, check_idx-2), min(N, check_idx+3)):
                        if ctx and ctx in norm_orig[w]:
                            window_hits += 1
                    score += min(window_hits, 2)
        # penalize if lines too far apart (no continuous matching)
        if score > best_score:
            best_score = score
            best_pos = i

    # require minimal absolute score relative to context size
    min_required = max(1, int(len(ctx_norm) * 1))  # stricter count but weighted earlier
    if best_score >= min_required:
        return best_pos
    # fuzzy window match over joined context
    ctx_join = " || ".join(ctx_norm)
    n = len(ctx_norm)
    for i in range(max(0, len(original_lines) - n + 1)):
        window = " || ".join([normalize_for_search(l) for l in original_lines[i:i+n]])
        ratio = difflib.SequenceMatcher(None, ctx_join, window).ratio()
        if ratio > FUZZY_THRESHOLD:
            return i + n - 1
    return -1

def find_by_keyword_fallback(original_lines: List[str], code_block: List[str]) -> int:
    if not code_block:
        return -1
    tokens = []
    for line in code_block:
        s = strip_inline_comments(line).strip()
        if not s:
            continue
        m = re.match(r'\s*case\s+([A-Za-z0-9_]+)', s, flags=re.IGNORECASE)
        if m:
            tokens.append('case:' + m.group(1).lower())
        for w in re.findall(r'[A-Za-z_][A-Za-z0-9_]{2,}', s):
            tokens.append(w.lower())
    if not tokens:
        return -1
    pos_scores = {}
    norm_orig = [normalize_for_search(l) for l in original_lines]
    for i, orig_line in enumerate(original_lines):
        low = norm_orig[i]
        score = 0
        for t in tokens:
            if t.startswith('case:'):
                keyword = t.split(':', 1)[1]
                if re.search(r'\bcase\s+' + re.escape(keyword) + r'\b', orig_line, flags=re.IGNORECASE):
                    score += 4
            else:
                if re.search(r'\b' + re.escape(t) + r'\b', low):
                    score += 1
        if score:
            pos_scores[i] = score
    if not pos_scores:
        # try weaker substring matching
        for i, low in enumerate(norm_orig):
            cnt = sum(1 for t in set(tokens) if t in low)
            if cnt:
                pos_scores[i] = cnt
    if not pos_scores:
        return -1
    best_idx, best_score = max(pos_scores.items(), key=lambda x: x[1])
    if best_score >= 2:
        return best_idx
    return -1

# ----------------- Donor parsing (more formats) -----------------
PLACEHOLDERS = [r'\[\.\]', r'\[\.\.\.\]', r'/\*\s*\[\.\]\s*\*/', r'/\*\s*\[\.\.\.\]\s*\*/', r'\.\.\.']
PLACEHOLDER_RE = re.compile('|'.join(PLACEHOLDERS))

def parse_donor_blocks(text: str) -> List[Tuple[List[str], List[str]]]:
    lines = text.splitlines()
    blocks: List[Tuple[List[str], List[str]]] = []
    i = 0
    while i < len(lines):
        if PLACEHOLDER_RE.search(lines[i]):
            # context before
            ctx_start = max(0, i - CONTEXT_LINES)
            context_before = [lines[j] for j in range(ctx_start, i) if lines[j].strip() != ""]
            code_block: List[str] = []
            # capture trailing content on same line after marker
            after = PLACEHOLDER_RE.sub('', lines[i]).strip()
            if after:
                code_block.append(after)
            # capture following non-empty lines as code (preferred)
            k = i + 1
            while k < len(lines):
                if lines[k].strip() == "":
                    break
                if PLACEHOLDER_RE.search(lines[k]):
                    break
                code_block.append(lines[k])
                k += 1
            # if no code after marker, try to capture code **before** marker (backward)
            if not code_block:
                back_lines = []
                b = i - 1
                while b >= 0 and len(back_lines) < BACKWARD_CODE_LINES:
                    if PLACEHOLDER_RE.search(lines[b]):
                        break
                    if lines[b].strip() == "":
                        # stop at blank line
                        break
                    back_lines.append(lines[b])
                    b -= 1
                # we captured reversed order, restore normal order
                back_lines.reverse()
                # only take as code if we have at least 1 non-comment line resembling code
                valid = [ln for ln in back_lines if ln.strip() and not ln.strip().startswith('//')]
                if valid:
                    code_block = back_lines
            blocks.append((context_before, code_block))
            i = k
        else:
            i += 1
    return blocks

# ----------------- Merge core -----------------
def already_exists_nearby(out_lines: List[str], insert_idx: int, code_block: List[str], window: int = 8) -> bool:
    if not code_block:
        return False
    first_nonempty = None
    for l in code_block:
        if l.strip():
            first_nonempty = l.strip()
            break
    if not first_nonempty:
        return False
    start = max(0, insert_idx - window)
    end = min(len(out_lines), insert_idx + window)
    for i in range(start, end):
        if first_nonempty in out_lines[i]:
            return True
    return False

def merge_one_file(donor_path: str, orig_path: str, backup_dir: str) -> Tuple[int, int]:
    logger.info(f"Verarbeite Donor: {donor_path} -> Original: {orig_path}")
    donor_text, _don_enc = try_read_file(donor_path)
    if donor_text is None:
        logger.error(f"Donor {donor_path} konnte nicht gelesen werden.")
        return 0, 0
    orig_text, orig_enc = try_read_file(orig_path)
    if orig_text is None:
        logger.error(f"Original {orig_path} konnte nicht gelesen werden.")
        return 0, 0
    original_lines = orig_text.splitlines()
    blocks = parse_donor_blocks(donor_text)
    total_blocks = len(blocks)
    inserted = 0
    if total_blocks == 0:
        logger.info(f"Keine Platzhalter in {os.path.basename(donor_path)} gefunden.")
        return 0, 0
    try:
        make_backup(orig_path, backup_dir)
    except Exception as e:
        logger.exception(f"Backup fehlgeschlagen: {e}")
    out_lines = original_lines.copy()
    for idx, (context_before, code_block) in enumerate(blocks, start=1):
        logger.info(f"Block {idx}/{total_blocks}: Code-Länge: {len(code_block)} Zeilen")
        if not code_block:
            logger.warning("Block ohne Code; überspringe.")
            continue
        pos = find_insert_position_improved(out_lines, context_before)
        if pos == -1:
            logger.info("Kontext-basierte Suche fehlgeschlagen, versuche Keyword-Fallback...")
            pos = find_by_keyword_fallback(out_lines, code_block)
        if pos == -1:
            logger.warning("Keine passende Einfügestelle gefunden (Block wird übersprungen).")
            # print small context in debug
            if context_before:
                logger.debug("Context before (last 3): %s", context_before[-3:])
            continue
        insert_idx = pos + 1
        insert_idx = adjust_for_preprocessor(out_lines, insert_idx)
        # check if already exists nearby -> skip to avoid duplicate
        if already_exists_nearby(out_lines, insert_idx, code_block, window=8):
            logger.info("Code scheint bereits in der Nähe vorhanden -> übersprungen (vermeidet Duplikat).")
            continue
        ref_line = out_lines[pos] if pos < len(out_lines) else ""
        indented_block = preserve_indentation(ref_line, code_block)
        for i, line in enumerate(indented_block):
            out_lines.insert(insert_idx + i, line)
        logger.info(f"Block {idx}: eingefügt bei Zeile {insert_idx+1} (nach match pos {pos+1}).")
        inserted += 1
    if inserted > 0:
        new_text = "\n".join(out_lines) + "\n"
        try:
            write_file_with_encoding(orig_path, new_text, encoding=(orig_enc or "utf-8"))
            logger.info(f"{os.path.basename(orig_path)} erfolgreich aktualisiert. {inserted}/{total_blocks} Blöcke eingefügt.")
        except Exception as e:
            logger.exception(f"Fehler beim Schreiben von {orig_path}: {e}")
    else:
        logger.info(f"Keine Änderungen an {os.path.basename(orig_path)} vorgenommen.")
    return inserted, total_blocks

# ----------------- Runner -----------------
def run_merge(donor_arg: Optional[str], orig_arg: Optional[str], backup_arg: Optional[str], single_file: Optional[str]):
    donor_dir, orig_dir, backup_dir = auto_detect_layout(donor_arg, orig_arg, backup_arg)
    logger.info(f"Pfad-Auflösung:\n  donor:  {donor_dir}\n  orig:   {orig_dir}\n  backup: {backup_dir}")
    if not os.path.isdir(donor_dir):
        logger.error(f"Donor-Ordner nicht gefunden: {donor_dir}")
        return
    if not os.path.isdir(orig_dir):
        logger.error(f"Original-Ordner nicht gefunden: {orig_dir}")
        return
    os.makedirs(backup_dir, exist_ok=True)
    total_files = 0
    total_blocks = 0
    total_inserted = 0
    if single_file:
        donor_path = os.path.join(donor_dir, single_file)
        orig_path = os.path.join(orig_dir, single_file)
        if not os.path.isfile(donor_path):
            logger.error(f"Donor-Datei nicht gefunden: {donor_path}")
            return
        if not os.path.isfile(orig_path):
            logger.error(f"Original-Datei nicht gefunden: {orig_path}")
            return
        inserted, blocks = merge_one_file(donor_path, orig_path, backup_dir)
        total_files = 1
        total_blocks = blocks
        total_inserted = inserted
    else:
        for root, _, files in os.walk(donor_dir):
            for fname in files:
                if not fname.lower().endswith(SUPPORTED_EXTS):
                    continue
                donor_path = os.path.join(root, fname)
                orig_path = os.path.join(orig_dir, fname)
                if not os.path.isfile(orig_path):
                    logger.warning(f"Original-Datei für {fname} nicht gefunden in {orig_dir}. Überspringe.")
                    continue
                total_files += 1
                inserted, blocks = merge_one_file(donor_path, orig_path, backup_dir)
                total_blocks += blocks
                total_inserted += inserted
    logger.info(f"Fertig: Dateien verarbeitet: {total_files}. Blöcke insgesamt: {total_blocks}. Eingefügt: {total_inserted}.")

# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Donor -> Original Merger (Metin2 focused).")
    p.add_argument("--donor", default=None, help="Donor folder (optional)")
    p.add_argument("--orig", default=None, help="Original folder (optional)")
    p.add_argument("--backup", default=None, help="Backup folder (optional)")
    p.add_argument("--file", default=None, help="Optional single filename to merge (e.g. char_item.cpp)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        run_merge(args.donor, args.orig, args.backup, args.file)
    except Exception as e:
        logger.exception(f"Unbehandelter Fehler: {e}")
        sys.exit(2)
