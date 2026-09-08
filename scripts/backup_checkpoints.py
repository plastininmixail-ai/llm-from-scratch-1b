"""Backup скрипт: копирует важные чекпойнты в безопасное место.

Защищает от:
- Случайного удаления (force-kill)
- Power loss во время save
- Перезаписи
"""
import shutil
import sys
from pathlib import Path
from datetime import datetime


SOURCE_DIR = Path("checkpoints")
BACKUP_DIR = Path("checkpoints_backup")
MAX_BACKUPS = 5  # максимум чекпойнтов в backup


def get_checkpoint_size(path: Path) -> str:
    """Размер чекпойнта в human-readable."""
    if not path.exists():
        return "0 MB"
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    if size > 1e9:
        return f"{size/1e9:.2f} GB"
    if size > 1e6:
        return f"{size/1e6:.0f} MB"
    return f"{size/1e3:.0f} KB"


def backup_checkpoint(ckpt_dir: Path, dry_run: bool = True) -> bool:
    """Скопировать один чекпойнт в backup."""
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        return False

    # Не бэкапим v0_1b_1000 (большой, содержит много чекпойнтов)
    target = BACKUP_DIR / ckpt_dir.name

    if target.exists():
        print(f"  [skip] {ckpt_dir.name} already in backup")
        return False

    size = get_checkpoint_size(ckpt_dir)
    print(f"  [backup] {ckpt_dir.name} ({size}) -> {target}")

    if dry_run:
        print(f"    [dry-run] would copy to {target}")
        return True

    target.parent.mkdir(parents=True, exist_ok=True)

    # Копируем только важные файлы (best.pt + train_log.jsonl)
    files_to_copy = list(ckpt_dir.glob("best.pt")) + list(ckpt_dir.glob("train_log.jsonl"))
    if not files_to_copy:
        return False

    for f in files_to_copy:
        dest = target / f.name
        shutil.copy2(f, dest)
        print(f"    copied {f.name}")

    return True


def list_backups():
    """Список существующих бэкапов."""
    if not BACKUP_DIR.exists():
        return []
    return sorted([d for d in BACKUP_DIR.iterdir() if d.is_dir()],
                  key=lambda d: d.stat().st_mtime)


def cleanup_old_backups(keep: int = MAX_BACKUPS):
    """Удалить старые бэкапы, оставить только последние `keep`."""
    backups = list_backups()
    if len(backups) <= keep:
        return

    to_remove = backups[:-keep]
    for b in to_remove:
        print(f"  [cleanup] removing old backup: {b.name}")
        shutil.rmtree(b)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true",
                   help="Actually perform backup (default: dry-run)")
    p.add_argument("--only", help="Only backup specific checkpoint name")
    args = p.parse_args()

    print(f"Backup dir: {BACKUP_DIR.absolute()}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}\n")

    if not SOURCE_DIR.exists():
        print(f"Source dir not found: {SOURCE_DIR}")
        return

    # Найти чекпойнты для бэкапа
    candidates = []
    for ckpt_dir in SOURCE_DIR.iterdir():
        if not ckpt_dir.is_dir():
            continue
        if args.only and ckpt_dir.name != args.only:
            continue
        # Только если есть best.pt
        if (ckpt_dir / "best.pt").exists():
            candidates.append(ckpt_dir)

    if not candidates:
        print("No checkpoints to backup")
        return

    print(f"Found {len(candidates)} checkpoints:\n")
    backed_up = 0
    for ckpt_dir in candidates:
        if backup_checkpoint(ckpt_dir, dry_run=not args.execute):
            backed_up += 1

    print(f"\nTotal: {backed_up} checkpoints {'backed up' if args.execute else 'would be backed up'}")

    if args.execute:
        cleanup_old_backups()

    # Показать текущие бэкапы
    print(f"\nCurrent backups:")
    backups = list_backups()
    if not backups:
        print("  (none)")
    else:
        total_size = 0
        for b in backups:
            size = get_checkpoint_size(b)
            print(f"  {b.name} ({size})")
            # Estimate size
            sz = sum(f.stat().st_size for f in b.rglob("*") if f.is_file())
            total_size += sz
        print(f"\nTotal backup size: {total_size/1e9:.2f} GB")


if __name__ == "__main__":
    main()
