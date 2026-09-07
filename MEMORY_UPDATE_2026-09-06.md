# Memory update — 2026-09-06

User profile changes:
- Тема бота больше НЕ стоицизм (отменено 2026-09-05)
- НЕ писать "жду команду" — писать предложения (06.09.2026)
- val_loss ≠ качество генерации

Key technical insights:
- BPE load() метод в tokenizer/bpe.py имеет баг — vocab показывает 0
- Workaround: manual load в скриптах (см. scripts/test_v101.py)
- vocab 1500 → 6-11 sub-tokens на RU слово (главная причина мусора)
- BPE 3000 обучен 06.09 на 173K docs (~80 мин CPU)

Checkpoints лидерборд (val_loss, 06.09.2026):
- 🥇 v100-wiki-20k: 3.62 (Wiki pretrained 20K, 1500 vocab)
- 🥈 v94-sft-final: 3.62 (SFT, лучший для Bridge Bot)
- 🥉 v101-3k-seq512: 4.30 (vocab 3000, seq 512)
- v97-v99: переобучены на SEP/Arxiv формат

Источники скачаны 06.09.2026:
- Fast.ai 28MB (27 SFT пар)
- SEP 44 files, IEP 22, Arxiv 50
- Wiki API 19, OpenStax 110 (только мета, PDF 404)
- Misc tools 32, Wiki dump bz2 25GB → 94,835 статей

KB: ~12K записей (10K Wiki + 866 chat + 95 facts + 110 OpenStax + 50 Arxiv + 36 philosophy + 27 Fast.ai).

Bridge Bot v14 (proc_2ce3799e4d2b, PID 12072):
- v94 + 1342 hardcoded + KB + TF-IDF + classifier 389 + sentiment
- Постоянно падает с Conflict (зомби, запрет force-kill)
- Production logs → data/production_logs.jsonl
