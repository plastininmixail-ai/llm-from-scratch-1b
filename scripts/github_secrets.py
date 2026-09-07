"""GitHub Actions Secrets Manager.

Добавляет secrets в репозиторий через GitHub API.
Использует шифрование публичным ключом.
"""
import base64
import sys
from pathlib import Path
from typing import Optional

import requests
from nacl import encoding, public

sys.path.insert(0, ".")
from scripts.github_helper import get_github_token, load_env


def get_public_key(repo: str, token: str) -> tuple[str, str]:
    """Получить public key репозитория для шифрования secrets."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    r = requests.get(
        f"https://api.github.com/repos/{repo}/actions/secrets/public-key",
        headers=headers, timeout=10
    )
    r.raise_for_status()
    return r.json()["key_id"], r.json()["key"]


def encrypt_secret(public_key_b64: str, secret_value: str) -> str:
    """Зашифровать secret используя libsodium."""
    pk_bytes = base64.b64decode(public_key_b64)
    pk = public.PublicKey(pk_bytes)
    sealed = public.SealedBox(pk).encrypt(secret_value.encode("utf-8"))
    return base64.b64encode(sealed).decode("utf-8")


def set_secret(repo: str, name: str, value: str, token: str) -> bool:
    """Установить secret в репозитории."""
    key_id, public_key = get_public_key(repo, token)
    encrypted = encrypt_secret(public_key, value)

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    r = requests.put(
        f"https://api.github.com/repos/{repo}/actions/secrets/{name}",
        headers=headers,
        json={"encrypted_value": encrypted, "key_id": key_id},
        timeout=10
    )
    return r.status_code in (201, 204)


def list_secrets(repo: str, token: str) -> list:
    """Список существующих secrets (только имена, без значений)."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    r = requests.get(
        f"https://api.github.com/repos/{repo}/actions/secrets",
        headers=headers, timeout=10
    )
    r.raise_for_status()
    return [s["name"] for s in r.json().get("secrets", [])]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="plastininmixail-ai/llm-from-scratch-1b")
    p.add_argument("--name", help="Secret name")
    p.add_argument("--value", help="Secret value (or read from env)")
    p.add_argument("--env-var", help="Read value from environment variable")
    p.add_argument("--list", action="store_true", help="List existing secrets")
    args = p.parse_args()

    load_env()
    token = get_github_token()

    if not token:
        print("[error] GITHUB_TOKEN not found")
        sys.exit(1)

    if args.list:
        secrets = list_secrets(args.repo, token)
        print(f"Secrets in {args.repo}:")
        for s in secrets:
            print(f"  - {s}")
        sys.exit(0)

    if not args.name:
        print("[error] --name required")
        sys.exit(1)

    # Get value
    value = args.value
    if not value and args.env_var:
        value = os.environ.get(args.env_var)
    if not value:
        print(f"[error] --value or --env-var required")
        sys.exit(1)

    # Set
    success = set_secret(args.repo, args.name, value, token)
    if success:
        print(f"[ok] Secret {args.name} set in {args.repo}")
    else:
        print(f"[error] Failed to set secret")
        sys.exit(1)
