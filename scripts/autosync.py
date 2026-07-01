#!/usr/bin/env python3
"""
Auto-sync script: watches for file changes and auto-commits + pushes to GitHub.
Once a change is detected, waits for a quiet period (no new changes) before committing.
"""

import os
import sys
import time
import subprocess
import fnmatch
from datetime import datetime

# ============ CONFIG ============
WATCH_DIR = "/home3/nxkh91/projects/motionfix"
DEBOUNCE_SECONDS = 120  # wait 2 min after last change before committing
IGNORE_PATTERNS = [
    ".git/*",
    "__pycache__/*",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    "scripts/autosync.py",    # don't trigger on changes to this script itself
    "scripts/sync.sh",        # don't trigger on the launcher
    "*.swp",
    "*.swo",
    "*~",
    ".gitignore.swp",
    # Large data / output directories — skip to keep scans fast
    "checkpoints/**",
    "data/training/**",
    "data/test_inputs/**",
    "outputs/**",
    "*.npy",          # binary output files
    "logs/**",
    "*.log",          # log files in general
]
# ===============================


# Directory prefixes to skip — any dir whose name starts with one of these is
# pruned in-place so os.walk never descends into it.
IGNORE_DIR_PREFIXES = (
    ".git",
    "__pycache__",
    "checkpoints",
    "data",
    "outputs",
    "logs",
)


def _is_ignored_dir(dirname):
    """Check whether a directory should be skipped entirely."""
    return dirname.startswith(IGNORE_DIR_PREFIXES)


def get_all_files():
    """Return dict of {relpath: mtime} for non-ignored files."""
    files = {}
    for root, dirs, filenames in os.walk(WATCH_DIR):
        # Prune ignored directories in-place so os.walk never descends into them
        dirs[:] = [d for d in dirs if not _is_ignored_dir(d)]

        for fname in filenames:
            relpath = os.path.relpath(os.path.join(root, fname), WATCH_DIR)

            # Apply ignore patterns (only to files, dirs handled above)
            ignored = False
            for pat in IGNORE_PATTERNS:
                if fnmatch.fnmatch(relpath, pat) or fnmatch.fnmatch(fname, pat):
                    ignored = True
                    break
            if ignored:
                continue

            try:
                mtime = os.path.getmtime(os.path.join(root, fname))
                files[relpath] = mtime
            except OSError:
                continue

    return files


def git_add_all():
    """Stage all changes."""
    subprocess.run(
        ["git", "add", "-A"],
        cwd=WATCH_DIR,
        capture_output=True,
    )


def git_commit_and_push():
    """Commit and push. Returns True if there was something to commit."""
    # Check if there are staged changes
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=WATCH_DIR,
    )
    if result.returncode == 0:
        # No changes to commit
        return False

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Build a useful message: counts + up to 8 changed paths (A/M/D/R)
    stat = subprocess.run(
        ["git", "diff", "--cached", "--name-status"],
        cwd=WATCH_DIR, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    n_a = sum(1 for l in stat if l.startswith("A"))
    n_m = sum(1 for l in stat if l.startswith("M"))
    n_d = sum(1 for l in stat if l.startswith("D"))
    n_r = sum(1 for l in stat if l.startswith("R"))
    summary = f"+{n_a} ~{n_m} -{n_d}" + (f" R{n_r}" if n_r else "")
    paths = [l.split("\t")[-1] for l in stat[:8]]
    body = "\n".join(paths) + ("\n…" if len(stat) > 8 else "")
    message = f"auto-sync ({summary}) {timestamp}\n\n{body}"

    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=WATCH_DIR,
        capture_output=True,
    )
    # explicit push (post-commit auto-push hook is disabled)
    subprocess.run(
        ["git", "push", "origin", "main"],
        cwd=WATCH_DIR,
        capture_output=True,
    )
    return True


def main():
    print(f"🔍 Watching for changes in: {WATCH_DIR}")
    print(f"⏱  Debounce: {DEBOUNCE_SECONDS}s (commit after no changes for this period)")
    print(f"📝 Modify a file to trigger auto-sync...")
    sys.stdout.flush()

    last_state = get_all_files()
    last_change_time = None

    while True:
        time.sleep(10)  # check every 10 seconds

        current_state = get_all_files()
        changed = False

        # Check for new or modified files
        for path, mtime in current_state.items():
            if path not in last_state or last_state[path] != mtime:
                changed = True
                break

        # Check for deleted files
        for path in last_state:
            if path not in current_state:
                changed = True
                break

        if changed:
            last_change_time = time.time()
            now_str = datetime.now().strftime("%H:%M:%S")
            print(f"[{now_str}] 🔔 Change detected, waiting {DEBOUNCE_SECONDS}s...")
            sys.stdout.flush()

        last_state = current_state

        # If enough time has passed since last change, commit & push
        if last_change_time is not None:
            elapsed = time.time() - last_change_time
            if elapsed >= DEBOUNCE_SECONDS:
                now_str = datetime.now().strftime("%H:%M:%S")
                print(f"[{now_str}] 📦 Committing & pushing...")
                sys.stdout.flush()

                git_add_all()
                if git_commit_and_push():
                    print(f"[{now_str}] ✅ Pushed to GitHub!")
                else:
                    print(f"[{now_str}] ⚠️  No changes to commit.")

                last_change_time = None
                sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Auto-sync stopped.")
