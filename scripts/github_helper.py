"""GitHub helper — читает GITHUB_TOKEN из .env, никогда не выводит."""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_env():
    """Загружает .env из текущей директории."""
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)


def get_github_token() -> Optional[str]:
    """Получить GitHub token из .env. Никогда не выводить!"""
    load_env()
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("[error] GITHUB_TOKEN not found in .env")
        return None
    return token


def get_github_headers() -> dict:
    """Headers для GitHub API с токеном."""
    token = get_github_token()
    if not token:
        return {}
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def check_github_auth() -> dict:
    """Проверить что токен работает. Возвращает user info."""
    import requests

    token = get_github_token()
    if not token:
        return {"authenticated": False, "error": "No token"}

    headers = get_github_headers()
    r = requests.get("https://api.github.com/user", headers=headers, timeout=10)

    if r.status_code == 200:
        user = r.json()
        # НЕ выводим токен, только user info
        return {
            "authenticated": True,
            "login": user.get("login"),
            "name": user.get("name"),
            "id": user.get("id"),
            "public_repos": user.get("public_repos"),
        }
    else:
        return {
            "authenticated": False,
            "status": r.status_code,
            "error": r.json().get("message", "Unknown error"),
        }


def list_repos(per_page: int = 10) -> list:
    """Список репозиториев пользователя."""
    import requests

    headers = get_github_headers()
    if not headers:
        return []

    r = requests.get(
        f"https://api.github.com/user/repos?per_page={per_page}&sort=updated",
        headers=headers, timeout=10
    )
    if r.status_code == 200:
        repos = r.json()
        return [{
            "name": repo["name"],
            "full_name": repo["full_name"],
            "private": repo["private"],
            "size_kb": repo["size"],
            "default_branch": repo["default_branch"],
        } for repo in repos]
    return []


def create_repo(name: str, description: str = "", private: bool = True) -> dict:
    """Создать новый репозиторий."""
    import requests

    headers = get_github_headers()
    if not headers:
        return {"error": "No token"}

    data = {
        "name": name,
        "description": description,
        "private": private,
        "auto_init": True,
    }
    r = requests.post(
        "https://api.github.com/user/repos",
        headers=headers, json=data, timeout=10
    )
    if r.status_code == 201:
        repo = r.json()
        return {
            "name": repo["name"],
            "full_name": repo["full_name"],
            "clone_url": repo["clone_url"],
            "html_url": repo["html_url"],
        }
    else:
        return {
            "error": r.status_code,
            "message": r.json().get("message", "Unknown")
        }


if __name__ == "__main__":
    # Проверка auth
    print("Checking GitHub authentication...")
    result = check_github_auth()
    if result.get("authenticated"):
        print(f"✓ Authenticated as: {result['login']}")
        print(f"  Name: {result.get('name')}")
        print(f"  Public repos: {result.get('public_repos')}")

        print("\nRecent repos:")
        repos = list_repos(per_page=5)
        for repo in repos:
            visibility = "private" if repo["private"] else "public"
            print(f"  - {repo['full_name']} ({visibility})")
    else:
        print(f"✗ Auth failed: {result.get('error')}")
