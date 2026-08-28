"""
HTTP-сервер для генерации текста через обученную модель.

Запуск:
    python -m inference.server --checkpoint checkpoints/medium-v2/best.pt --port 8765

Запрос:
    POST /generate
    {"prompt": "The capital of France is",
     "max_new_tokens": 50, "temperature": 0.8,
     "top_k": 40, "top_p": 0.95}

Ответ:
    {"text": "...", "tokens_generated": 50, "time_sec": 1.23}
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from inference.generate import generate_text, load_model_from_checkpoint


class GenerationHandler(BaseHTTPRequestHandler):
    model = None  # type: ignore
    tokenizer = None  # type: ignore
    device = "cpu"

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        # Тише, чем stderr по умолчанию
        print(f"[{self.log_date_time_string()}] {format % args}")

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "checkpoint": self._checkpoint_name()})
        elif self.path == "/" or self.path == "/info":
            self._send_json(
                200,
                {
                    "model": "gpt-from-scratch",
                    "checkpoint": self._checkpoint_name(),
                    "device": self.device,
                    "endpoints": ["GET /health", "POST /generate"],
                },
            )
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/generate":
            self._send_json(404, {"error": "unknown endpoint"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._send_json(400, {"error": f"invalid json: {e}"})
            return

        prompt = payload.get("prompt", "")
        if not prompt:
            self._send_json(400, {"error": "missing 'prompt'"})
            return

        max_new_tokens = int(payload.get("max_new_tokens", 50))
        temperature = float(payload.get("temperature", 0.8))
        top_k = int(payload.get("top_k", 40))
        top_p = float(payload.get("top_p", 0.95))

        t0 = time.time()
        try:
            text = generate_text(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=self.device,
            )
        except Exception as e:
            self._send_json(500, {"error": f"generation failed: {e}"})
            return
        elapsed = time.time() - t0

        self._send_json(
            200,
            {
                "text": text,
                "prompt": prompt,
                "tokens_generated": max_new_tokens,
                "time_sec": round(elapsed, 3),
                "tokens_per_sec": round(max_new_tokens / max(elapsed, 1e-6), 1),
            },
        )

    # helpers ----------------------------------------------------------------
    def _send_json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _checkpoint_name() -> str:
        return getattr(GenerationHandler, "_ckpt_path", "?")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer/vocab.json"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    print(f"=== gpt-from-scratch HTTP сервер ===")
    print(f"загружаю {args.checkpoint}…")
    model, tokenizer, _ = load_model_from_checkpoint(args.checkpoint, args.tokenizer)
    print(f"✓ модель загружена")

    GenerationHandler.model = model
    GenerationHandler.tokenizer = tokenizer
    GenerationHandler._ckpt_path = str(args.checkpoint)

    httpd = ThreadingHTTPServer((args.host, args.port), GenerationHandler)
    print(f"✓ слушаю на http://{args.host}:{args.port}/")
    print(f"  POST /generate  {{\"prompt\": \"...\", \"max_new_tokens\": 50}}")
    print(f"  GET  /health")
    print("Ctrl+C для остановки")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nостановлено")
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
