"""Скачивание дополнительных данных для Stage 0/1.

Источники:
1. Wikipedia RU (через HF datasets)
2. Wikipedia multilingual (subset)
3. GitHub code corpus (через GH API)
4. The Stack (код, открытый)

Usage:
    python -m scripts.download_extra_data --wiki-ru
    python -m scripts.download_extra_data --github-code --limit 1000
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, ".")


def download_wiki_ru(max_size_gb: float = 5.0):
    """Скачать Wikipedia RU через HF datasets."""
    print(f"\n=== Wikipedia RU ({max_size_gb} GB) ===")
    try:
        from datasets import load_dataset

        out_dir = Path("data/sources/wiki_ru")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Wikimedia/wikipedia датасет с русским
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.ru",
            split="train",
            streaming=True,
        )

        count = 0
        size_bytes = 0
        max_bytes = int(max_size_gb * 1e9)
        out_file = out_dir / "wiki_ru.jsonl"

        with out_file.open("w", encoding="utf-8") as f:
            for row in ds:
                text = row.get("text", "")
                if len(text) < 100:
                    continue
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                count += 1
                size_bytes += len(text.encode("utf-8"))
                if size_bytes >= max_bytes:
                    break
                if count % 10000 == 0:
                    print(f"  {count} articles, {size_bytes/1e9:.2f} GB")

        print(f"Done: {count} articles, {size_bytes/1e9:.2f} GB")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def download_wikipedia_sample(max_size_gb: float = 10.0, languages: list = None):
    """Скачать Wikipedia для нескольких языков."""
    if languages is None:
        languages = ["ru", "en", "de", "fr", "es", "zh"]

    print(f"\n=== Wikipedia multilingual ({languages}) ===")
    try:
        from datasets import load_dataset

        out_dir = Path("data/sources/wiki_multi")
        out_dir.mkdir(parents=True, exist_ok=True)

        total_size = 0
        max_bytes = int(max_size_gb * 1e9)

        for lang in languages:
            if total_size >= max_bytes:
                break

            print(f"\n  [{lang}] downloading...")
            try:
                ds = load_dataset(
                    "wikimedia/wikipedia",
                    f"20231101.{lang}",
                    split="train",
                    streaming=True,
                )

                count = 0
                lang_size = 0
                out_file = out_dir / f"wiki_{lang}.jsonl"

                with out_file.open("w", encoding="utf-8") as f:
                    for row in ds:
                        text = row.get("text", "")
                        if len(text) < 100:
                            continue
                        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        count += 1
                        lang_size += len(text.encode("utf-8"))
                        total_size += len(text.encode("utf-8"))

                        if count >= 50000 or total_size >= max_bytes:
                            break

                print(f"  [{lang}] {count} articles, {lang_size/1e6:.1f} MB")
            except Exception as e:
                print(f"  [{lang}] ERROR: {e}")

        print(f"\nTotal: {total_size/1e9:.2f} GB")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def download_github_code(token: str, limit: int = 1000, languages: list = None):
    """Скачать GitHub code через API."""
    if languages is None:
        languages = ["python", "javascript", "typescript", "go", "rust", "java"]

    if not token:
        print("ERROR: GITHUB_TOKEN required")
        return False

    print(f"\n=== GitHub Code ({limit} files, langs={languages}) ===")

    import requests
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    out_dir = Path("data/sources/github_code")
    out_dir.mkdir(parents=True, exist_ok=True)

    for lang in languages:
        print(f"\n  [{lang}] searching code...")
        try:
            # Search for popular repos with code in this language
            r = requests.get(
                "https://api.github.com/search/code",
                params={
                    "q": f"language:{lang} stars:>100",
                    "sort": "stars",
                    "per_page": min(limit // len(languages), 30),
                },
                headers=headers,
                timeout=15,
            )
            r.raise_for_status()
            results = r.json().get("items", [])

            count = 0
            for item in results[:limit // len(languages)]:
                # Get raw content via blob URL
                repo_full_name = item["repository"]["full_name"]
                file_path = item["path"]

                # Get default branch
                repo_info = requests.get(
                    f"https://api.github.com/repos/{repo_full_name}",
                    headers=headers, timeout=10
                ).json()
                default_branch = repo_info.get("default_branch", "main")

                raw_url = f"https://raw.githubusercontent.com/{repo_full_name}/{default_branch}/{file_path}"
                content_r = requests.get(raw_url, timeout=15)
                if content_r.status_code == 200:
                    content = content_r.text
                    if len(content) < 50 or len(content) > 50000:
                        continue

                    out_file = out_dir / f"{lang}_{count:04d}.txt"
                    out_file.write_text(content, encoding="utf-8", errors="ignore")
                    count += 1

            print(f"  [{lang}] {count} files saved")
        except Exception as e:
            print(f"  [{lang}] ERROR: {e}")

    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wiki-ru", action="store_true", help="Download Russian Wikipedia")
    p.add_argument("--wiki-multi", action="store_true", help="Download multi-lang Wikipedia")
    p.add_argument("--github-code", action="store_true", help="Download GitHub code")
    p.add_argument("--wiki-ru-size", type=float, default=5.0)
    p.add_argument("--wiki-multi-size", type=float, default=10.0)
    p.add_argument("--github-limit", type=int, default=200)
    args = p.parse_args()

    if not any([args.wiki_ru, args.wiki_multi, args.github_code]):
        print("Specify at least one: --wiki-ru, --wiki-multi, --github-code")
        return

    if args.wiki_ru:
        download_wiki_ru(args.wiki_ru_size)

    if args.wiki_multi:
        download_wikipedia_sample(args.wiki_multi_size)

    if args.github_code:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        token = os.environ.get("GITHUB_TOKEN")
        download_github_code(token, args.github_limit)


if __name__ == "__main__":
    main()
