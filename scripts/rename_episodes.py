"""Rename FLV episodes like 'אווטר פרק 1 עי_יהודה.FLV' to '1.FLV'.

Usage:
  python scripts/rename_episodes.py FOLDER            # show mapping and checks only
  python scripts/rename_episodes.py FOLDER --execute  # rename if every check passes
  python scripts/rename_episodes.py --undo LOG.json   # reverse the renames recorded in a log
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "rename_logs"
TARGET_EXT = ".FLV"
# The only naming pattern found in the folder; episode number follows the word פרק ("episode").
SOURCE_RE = re.compile(r"^אווטר פרק (\d+) עי_יהודה\.flv$", re.IGNORECASE)
TARGET_RE = re.compile(r"^[1-9]\d*\.FLV$")


def build_plan(folder: Path) -> tuple[list[dict], list[str]]:
    entries = sorted(folder.iterdir(), key=lambda p: p.name)
    flvs = [p for p in entries if p.is_file() and p.suffix.lower() == ".flv"]
    all_names_lower = {p.name.lower(): p.name for p in entries}
    plan, problems = [], []
    for p in flvs:
        name = p.name
        if TARGET_RE.match(name):
            plan.append({"old": name, "new": name, "episode": int(name[:-4]), "action": "unchanged"})
            continue
        matches = re.findall(r"פרק\s*(\d+)", name)
        m = SOURCE_RE.match(name)
        if not m or len(matches) != 1:
            problems.append(f"{name!r}: does not match the expected pattern "
                            f"'אווטר פרק <N> עי_יהודה.FLV' (found {len(matches)} 'פרק <N>' occurrences)")
            continue
        ep = int(m.group(1))
        if ep <= 0:
            problems.append(f"{name!r}: episode number {m.group(1)} is not a positive number")
            continue
        plan.append({"old": name, "new": f"{ep}{TARGET_EXT}", "episode": ep, "action": "rename"})

    by_ep: dict[int, list[str]] = {}
    for e in plan:
        by_ep.setdefault(e["episode"], []).append(e["old"])
    for ep, names in sorted(by_ep.items()):
        if len(names) > 1:
            problems.append(f"episode {ep} claimed by several files: {names}")

    sources_lower = {e["old"].lower() for e in plan}
    targets: dict[str, str] = {}
    for e in plan:
        tl = e["new"].lower()
        if tl in targets and targets[tl] != e["old"]:
            problems.append(f"{e['old']!r} and {targets[tl]!r} would both become {e['new']!r}")
        targets[tl] = e["old"]
        if e["action"] == "rename" and tl in all_names_lower and tl != e["old"].lower():
            existing = all_names_lower[tl]
            if tl not in sources_lower or existing != e["old"]:
                problems.append(f"{e['old']!r} -> {e['new']!r}: destination already exists as {existing!r}")

    if plan:
        eps = sorted(by_ep)
        gaps = sorted(set(range(1, eps[-1] + 1)) - set(eps))
        if gaps:
            problems.append(f"episode numbers missing from the sequence 1..{eps[-1]}: {gaps}")
    plan.sort(key=lambda e: e["episode"])
    return plan, problems


def execute(folder: Path, plan: list[dict]) -> dict:
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{folder.name}_undo_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
    log = {"folder": str(folder), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "planned": [{"old": e["old"], "new": e["new"]} for e in plan if e["action"] == "rename"],
           "succeeded": [], "failed": []}

    def save() -> None:
        tmp = log_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, log_path)

    save()
    print(f"Undo log: {log_path}")
    for e in plan:
        if e["action"] != "rename":
            continue
        src, dst = folder / e["old"], folder / e["new"]
        try:
            if dst.exists():
                raise FileExistsError(f"destination appeared: {dst.name}")
            src.rename(dst)  # Windows rename never replaces an existing file
            log["succeeded"].append({"old": e["old"], "new": e["new"]})
        except OSError as exc:
            log["failed"].append({"old": e["old"], "new": e["new"], "error": str(exc)})
        save()
    return log


def undo(log_path: Path) -> int:
    log = json.loads(log_path.read_text(encoding="utf-8"))
    folder = Path(log["folder"])
    failed = 0
    for e in reversed(log["succeeded"]):
        src, dst = folder / e["new"], folder / e["old"]
        if dst.exists() or not src.exists():
            print(f"skip {e['new']!r} -> {e['old']!r} (source missing or destination exists)")
            failed += 1
            continue
        src.rename(dst)
        print(f"restored {e['new']!r} -> {e['old']!r}")
    return 1 if failed else 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", type=Path)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--undo", type=Path)
    a = ap.parse_args()
    if a.undo:
        return undo(a.undo)
    folder = a.folder.resolve()
    plan, problems = build_plan(folder)
    print(f"Folder: {folder}")
    print("Mapping (old -> new):")
    for e in plan:
        print(f"  {e['old']}  ->  {e['new']}" + ("  (unchanged)" if e["action"] == "unchanged" else ""))
    print(f"{len(plan)} FLV files; {sum(e['action'] == 'rename' for e in plan)} to rename")
    if problems:
        print("PROBLEMS - no changes made:")
        for p in problems:
            print("  - " + p)
        return 2
    print("Checks: all episode numbers present and unique, no destination collisions (case-insensitive).")
    if not a.execute:
        return 0
    log = execute(folder, plan)
    final = sorted((p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".flv"),
                   key=lambda n: (int(n[:-4]) if TARGET_RE.match(n) else 10**9, n))
    print("Resulting FLV files:")
    for n in final:
        print("  " + n)
    unchanged = sum(e["action"] == "unchanged" for e in plan)
    print(f"renamed={len(log['succeeded'])} unchanged={unchanged} failed={len(log['failed'])}")
    for f in log["failed"]:
        print(f"  FAILED {f['old']!r} -> {f['new']!r}: {f['error']}")
    return 1 if log["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
