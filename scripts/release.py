#!/usr/bin/env python3
"""
Release Script - MBO Trading Strategy Analyzer
Automatikus verzió frissítés, backup és GitHub push.

Használat:
    python scripts/release.py 5.4.3
    python scripts/release.py 5.5.0 --message "Custom commit message"
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Projekt gyökér mappa
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
BACKUPS_DIR = PROJECT_ROOT / "backups"
VERSION_FILE = SRC_DIR / "version.txt"
INIT_FILE = SRC_DIR / "__init__.py"


def parse_args():
    """Argumentumok feldolgozása."""
    parser = argparse.ArgumentParser(
        description="Release script - verzió frissítés, backup, GitHub push"
    )
    parser.add_argument(
        "version",
        help="Új verzió szám (pl. 5.4.3)"
    )
    parser.add_argument(
        "-m", "--message",
        default=None,
        help="Egyedi commit üzenet (opcionális)"
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Ne pusholjon GitHub-ra"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Ne készítsen backup-ot"
    )
    return parser.parse_args()


def validate_version(version: str) -> bool:
    """Verzió formátum ellenőrzése."""
    pattern = r"^\d+\.\d+\.\d+$"
    return bool(re.match(pattern, version))


def get_current_version() -> str:
    """Aktuális verzió lekérdezése."""
    if INIT_FILE.exists():
        content = INIT_FILE.read_text(encoding="utf-8")
        match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
        if match:
            return match.group(1)
    return "unknown"


def update_init_file(version: str) -> None:
    """__init__.py verzió frissítése."""
    content = INIT_FILE.read_text(encoding="utf-8")
    new_content = re.sub(
        r'__version__\s*=\s*"[^"]+"',
        f'__version__ = "{version}"',
        content
    )
    INIT_FILE.write_text(new_content, encoding="utf-8")
    print(f"[OK] {INIT_FILE.name} frissítve: {version}")


def update_version_txt(version: str, changelog: str) -> None:
    """version.txt frissítése."""
    old_content = VERSION_FILE.read_text(encoding="utf-8") if VERSION_FILE.exists() else ""

    new_entry = f"v{version} Stable\n{changelog}\n\n"
    new_content = new_entry + old_content

    VERSION_FILE.write_text(new_content, encoding="utf-8")
    print(f"[OK] {VERSION_FILE.name} frissítve")


def create_backup(version: str) -> Path:
    """Backup készítése az src mappáról."""
    BACKUPS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"src_v{version}_{timestamp}.zip"
    backup_path = BACKUPS_DIR / backup_name

    # ZIP készítése
    shutil.make_archive(
        str(backup_path).replace(".zip", ""),
        "zip",
        SRC_DIR
    )

    # Méret lekérdezése
    size_mb = backup_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Backup: {backup_name} ({size_mb:.2f} MB)")

    return backup_path


def git_add_commit_push(version: str, message: str, push: bool) -> None:
    """Git műveletek végrehajtása."""
    os.chdir(PROJECT_ROOT)

    # Stage changes
    subprocess.run(["git", "add", "-A"], check=True, capture_output=True)
    print("[OK] Git: változások hozzáadva")

    # Commit
    full_message = f"{message}\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
    subprocess.run(
        ["git", "commit", "-m", full_message],
        check=True,
        capture_output=True
    )
    print(f"[OK] Git: commit létrehozva")

    # Push
    if push:
        result = subprocess.run(
            ["git", "push"],
            check=True,
            capture_output=True,
            text=True
        )
        print("[OK] Git: push sikeres")
    else:
        print("[SKIP] Git push kihagyva (--no-push)")


def get_git_changes() -> str:
    """Git változások összefoglalása a changelog-hoz."""
    os.chdir(PROJECT_ROOT)

    result = subprocess.run(
        ["git", "diff", "--stat", "--cached"],
        capture_output=True,
        text=True
    )

    if not result.stdout.strip():
        # Ha nincs staged, nézzük az unstaged változásokat
        result = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True,
            text=True
        )

    return result.stdout.strip()


def main():
    """Fő függvény."""
    args = parse_args()

    # Verzió validálás
    if not validate_version(args.version):
        print(f"[HIBA] Érvénytelen verzió formátum: {args.version}")
        print("       Helyes formátum: X.Y.Z (pl. 5.4.3)")
        sys.exit(1)

    current = get_current_version()
    print(f"\n{'='*50}")
    print(f"  RELEASE: v{current} -> v{args.version}")
    print(f"{'='*50}\n")

    # 1. Verzió frissítés
    update_init_file(args.version)

    # Changelog (alapértelmezett vagy egyedi)
    if args.message:
        changelog = f"- {args.message}"
    else:
        changelog = "- Release"

    update_version_txt(args.version, changelog)

    # 2. Backup
    if not args.no_backup:
        create_backup(args.version)
    else:
        print("[SKIP] Backup kihagyva (--no-backup)")

    # 3. Git commit + push
    commit_message = args.message or f"v{args.version} Stable"
    try:
        git_add_commit_push(args.version, commit_message, not args.no_push)
    except subprocess.CalledProcessError as e:
        print(f"[HIBA] Git művelet sikertelen: {e}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  KÉSZ! v{args.version} Stable")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
