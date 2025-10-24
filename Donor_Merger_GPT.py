#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Donor_Merger_GPT.py
Final: Adds --force and --show-skipped to control duplicate behavior and debug skipped blocks.
Place next to donor/ original/ backup and run in your venv.
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
from typing import List, Tuple, Optional, Dict, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
CONTEXT_LINES = 8
BACKWARD_CODE_LINES = 8
FUZZY_THRESHOLD = 0.55
ENCODINGS_TRY = ["utf-8", "cp949", "euc-kr", "latin-1", "cp1252"]
SUPPORTED_EXTS = (".cpp", ".c", ".h", ".hpp", ".txt")
PLACEHOLDERS = [r'\[\.\]', r'\[\.\.\.\]', r'/\*\s*\[\.\]\s*\*/', r'/\*\s*\[\.\.\.\]\s*\*/', r'\.\.\.']
PLACEHOLDER_RE = re.compile('|'.join(PLACEHOLDERS))

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

def write_atomic(path: str, text: str, encoding: str = "utf-8"):
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

# ----------------- Donor Parsing -----------------
def strip_blank_ends(lst: List[str]) -> List[str]:
    while lst and (lst[0].strip() == ""):
        lst.pop(0)
    while lst and (lst[-1].strip() == ""):
        lst.pop()
    return lst

def parse_donor_blocks(text: str) -> List[Tuple[List[str], List[str]]]:
    lines = text.splitlines()
    blocks: List[Tuple[List[str], List[str]]] = []
    i = 0
    N = len(lines)
    while i < N:
        if PLACEHOLDER_RE.search(lines[i]):
            ctx_start = max(0, i - CONTEXT_LINES)
            context_before = [lines[j] for j in range(ctx_start, i) if lines[j].strip() != ""]
            code_block: List[str] = []
            after = PLACEHOLDER_RE.sub('', lines[i]).strip()
            if after:
                code_block.append(after)
            paired_idx = None
            for k in range(i+1, min(N, i+300)):
                if PLACEHOLDER_RE.search(lines[k]):
                    paired_idx = k
                    break
            if paired_idx:
                for kk in range(i+1, paired_idx):
                    code_block.append(lines[kk])
                blocks.append((context_before, strip_blank_ends(code_block.copy())))
                i = paired_idx + 1
                continue
            k = i + 1
            while k < N:
                if lines[k].strip() == "":
                    break
                if PLACEHOLDER_RE.search(lines[k]):
                    break
                code_block.append(lines[k])
                k += 1
            if not any(ln.strip() for ln in code_block):
                back = []
                b = i - 1
                while b >= 0 and len(back) < BACKWARD_CODE_LINES:
                    if PLACEHOLDER_RE.search(lines[b]):
                        break
                    if lines[b].strip() == "":
                        break
                    back.append(lines[b])
                    b -= 1
                back.reverse()
                if back:
                    code_block = back
            blocks.append((context_before, strip_blank_ends(code_block.copy())))
            i = k
        else:
            i += 1
    return blocks

# ----------------- Matching -----------------
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
    N = len(original_lines)
    for i in range(N):
        score = 0
        for offset, ctx in enumerate(reversed(ctx_norm)):
            check_idx = i - offset
            if check_idx < 0:
                break
            pat = build_flexible_regex_from_line(ctx)
            if re.search(pat, original_lines[check_idx], flags=re.IGNORECASE):
                score += 3
            else:
                if ctx and ctx in norm_orig[check_idx]:
                    score += 1
                else:
                    hits = sum(1 for w in range(max(0, check_idx-2), min(N, check_idx+3)) if ctx and ctx in norm_orig[w])
                    score += min(hits, 2)
        if score > best_score:
            best_score = score
            best_pos = i
    min_required = max(1, int(len(ctx_norm) * 1))
    if best_score >= min_required:
        return best_pos
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

# ----------------- Duplicate detection (returns details) -----------------
def find_duplicate_nearby(out_lines: List[str], insert_idx: int, code_block: List[str], window: int = 8) -> Dict[str, Any]:
    """
    Returns dict:
      { 'exists': bool, 'reason': str, 'matches': [ (line_idx, snippet) ... ] }
    """
    nonempty = [l for l in code_block if l.strip()]
    result = {'exists': False, 'reason': '', 'matches': []}
    if not nonempty:
        return result
    N = len(out_lines)
    start = max(0, insert_idx - window)
    end = min(N, insert_idx + window)
    if len(nonempty) == 1:
        target = nonempty[0].strip()
        for i in range(start, end):
            if out_lines[i].strip() == target:
                result['exists'] = True
                result['reason'] = 'single-line exact match'
                result['matches'].append((i, out_lines[i].strip()))
                return result
        # try substring hits for info
        for i in range(start, end):
            if target in out_lines[i]:
                result['matches'].append((i, out_lines[i].strip()))
        return result
    # multi-line: search for 2-line consecutive exact match
    seqs = []
    for i in range(len(nonempty) - 1):
        seqs.append((nonempty[i].strip(), nonempty[i+1].strip()))
    for i in range(start, max(start, end-1)):
        a = out_lines[i].strip()
        b = out_lines[i+1].strip() if i+1 < len(out_lines) else ""
        for x, y in seqs:
            if a == x and b == y:
                result['exists'] = True
                result['reason'] = 'multi-line consecutive exact match'
                result['matches'].append((i, f"{a} | {b}"))
                return result
    # not exact; collect some substring hits as info
    for i in range(start, end):
        low = out_lines[i].lower()
        hits = sum(1 for t in set([w.lower() for w in nonempty]) if t and t in low)
        if hits:
            result['matches'].append((i, out_lines[i].strip()))
    return result

# ----------------- Merge core -----------------
def merge_one_file(donor_path: str, orig_path: str, backup_dir: str, dry_run: bool=False, verbose: bool=False, force: bool=False, show_skipped: bool=False) -> Tuple[int, int]:
    logger.info(f"Verarbeite Donor: {donor_path} -> Original: {orig_path}")
    donor_text, _ = try_read_file(donor_path)
    orig_text, orig_enc = try_read_file(orig_path)
    if donor_text is None or orig_text is None:
        logger.error("Lesefehler; überspringe Datei.")
        return 0, 0
    blocks = parse_donor_blocks(donor_text)
    total_blocks = len(blocks)
    inserted = 0
    if total_blocks == 0:
        logger.info("Keine Platzhalter im Donor gefunden.")
        return 0, 0
    if not dry_run:
        make_backup(orig_path, backup_dir)
    out_lines = orig_text.splitlines()
    for idx, (context_before, code_block) in enumerate(blocks, start=1):
        logger.info(f"Block {idx}/{total_blocks}: Code-Länge: {len(code_block)} Zeilen")
        if not code_block:
            logger.warning("Block ohne Code; überspringe.")
            continue
        pos = find_insert_position_improved(out_lines, context_before)
        used_fallback = False
        if pos == -1:
            if verbose: logger.info("Kontext-Matching fehlgeschlagen -> Keyword-Fallback")
            pos = find_by_keyword_fallback(out_lines, code_block)
            used_fallback = True
        if pos == -1:
            if show_skipped:
                logger.warning("Keine Einfügestelle gefunden — zeige Donor-Block (Vorschau):")
                for ln in (code_block[:6] if len(code_block)>6 else code_block):
                    logger.warning("  %s", ln.rstrip())
            logger.warning("Block übersprungen (kein Match).")
            continue
        insert_idx = pos + 1
        insert_idx = adjust_for_preprocessor(out_lines, insert_idx)
        dup_info = find_duplicate_nearby(out_lines, insert_idx, code_block, window=8)
        if dup_info['exists'] and not force:
            logger.info("Code scheint bereits in der Nähe vorhanden -> übersprungen (Duplikatschutz).")
            if show_skipped:
                logger.info("Duplikat-Details: reason=%s matches=%s", dup_info['reason'], dup_info['matches'][:5])
                logger.info("Donor-Block (erste 6 Zeilen):")
                for ln in (code_block[:6] if len(code_block)>6 else code_block):
                    logger.info("  %s", ln.rstrip())
                # show nearby original context
                for mi, snippet in dup_info['matches'][:6]:
                    start = max(0, mi-3)
                    end = min(len(out_lines), mi+4)
                    logger.info(" Original-Lines around %d:", mi+1)
                    for i in range(start, end):
                        logger.info("   %4d | %s", i+1, out_lines[i].rstrip())
            continue
        # if force and dup exists, still proceed but log
        if dup_info['exists'] and force:
            logger.warning("Force-Flag aktiv: Duplikat erkannt (%s) — erzwinge Einfügen.", dup_info.get('reason','?'))
        ref_line = out_lines[pos] if pos < len(out_lines) else ""
        indented_block = preserve_indentation(ref_line, code_block)
        if dry_run:
            logger.info(f"[dry-run] würde {len(indented_block)} Zeilen bei Zeile {insert_idx+1} einfügen (fallback={used_fallback})")
            if verbose:
                for ln in indented_block[:8]:
                    logger.info("  -> %s", ln.rstrip())
            inserted += 1
            continue
        for i, ln in enumerate(indented_block):
            out_lines.insert(insert_idx + i, ln)
        logger.info(f"Block {idx} eingefügt bei Zeile {insert_idx+1} ({len(indented_block)} Zeilen). (fallback={used_fallback})")
        if verbose:
            for ln in indented_block[:8]:
                logger.info("  -> %s", ln.rstrip())
        inserted += 1
    if inserted > 0 and not dry_run:
        new_text = "\n".join(out_lines) + "\n"
        write_atomic(orig_path, new_text, encoding=(orig_enc or "utf-8"))
        logger.info(f"{os.path.basename(orig_path)} aktualisiert: {inserted}/{total_blocks} Blöcke eingefügt.")
    elif inserted > 0 and dry_run:
        logger.info(f"[dry-run] würde {inserted}/{total_blocks} Blöcke einfügen.")
    else:
        logger.info("Keine Änderungen vorgenommen.")
    return inserted, total_blocks

# ----------------- Runner -----------------
def resolve_dirs(donor_arg, orig_arg, backup_arg):
    if donor_arg or orig_arg or backup_arg:
        d = os.path.abspath(donor_arg) if donor_arg else os.path.join(BASE_DIR, "donor")
        o = os.path.abspath(orig_arg) if orig_arg else os.path.join(BASE_DIR, "original")
        b = os.path.abspath(backup_arg) if backup_arg else os.path.join(BASE_DIR, "backup")
        return d,o,b
    local_d = os.path.join(BASE_DIR, "donor")
    local_o = os.path.join(BASE_DIR, "original")
    local_b = os.path.join(BASE_DIR, "backup")
    if os.path.isdir(local_d) and os.path.isdir(local_o):
        return local_d, local_o, local_b
    zut_d = os.path.join(BASE_DIR, "ZUTUN", "donor")
    zut_o = os.path.join(BASE_DIR, "ZUTUN", "original")
    zut_b = os.path.join(BASE_DIR, "ZUTUN", "backup")
    if os.path.isdir(zut_d) and os.path.isdir(zut_o):
        return zut_d, zut_o, zut_b
    return local_d, local_o, local_b

def collect_files(donor_dir: str) -> List[str]:
    files = []
    for root, _, fs in os.walk(donor_dir):
        for f in fs:
            if f.lower().endswith(SUPPORTED_EXTS):
                rel = os.path.relpath(os.path.join(root, f), donor_dir)
                files.append(rel)
    files.sort()
    return files

def parse_args():
    p = argparse.ArgumentParser(description="Donor -> Original Merger (Metin2)")
    p.add_argument("--donor", default=None)
    p.add_argument("--orig", default=None)
    p.add_argument("--backup", default=None)
    p.add_argument("--file", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--force", action="store_true", help="Force insertion even if duplicate detected")
    p.add_argument("--show-skipped", action="store_true", help="Show details about skipped blocks")
    return p.parse_args()

def main():
    args = parse_args()
    donor_dir, orig_dir, backup_dir = resolve_dirs(args.donor, args.orig, args.backup)
    logger.info(f"Pfad-Auflösung:\n  donor:  {donor_dir}\n  orig:   {orig_dir}\n  backup: {backup_dir}")
    if not os.path.isdir(donor_dir):
        logger.error(f"Donor-Ordner nicht gefunden: {donor_dir}")
        return
    if not os.path.isdir(orig_dir):
        logger.error(f"Original-Ordner nicht gefunden: {orig_dir}")
        return
    os.makedirs(backup_dir, exist_ok=True)
    if args.file:
        files = [args.file]
    else:
        files = collect_files(donor_dir)
    total_files = 0
    total_blocks = 0
    total_inserted = 0
    for fn in files:
        donor_path = os.path.join(donor_dir, fn)
        orig_path = os.path.join(orig_dir, fn)
        total_files += 1
        if not os.path.isfile(orig_path):
            logger.warning(f"Original-Datei für {fn} nicht gefunden: {orig_path} - übersprungen")
            continue
        ins, blocks = merge_one_file(donor_path, orig_path, backup_dir, dry_run=args.dry_run, verbose=args.verbose, force=args.force, show_skipped=args.show_skipped)
        total_blocks += blocks
        total_inserted += ins
    logger.info(f"Fertig: Dateien verarbeitet: {total_files}. Blöcke insgesamt: {total_blocks}. Eingefügt: {total_inserted}.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Unbehandelter Fehler: %s", e)
        sys.exit(2)
