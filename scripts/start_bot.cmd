@echo off
cd /d C:\Users\mixai\Desktop\llm-from-scratch
call .venv\Scripts\activate
python -m inference.telegram_bot --checkpoint checkpoints/chat-large-v55/best.pt --tokenizer tokenizer/vocab.json
