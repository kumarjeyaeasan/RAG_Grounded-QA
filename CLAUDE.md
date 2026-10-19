# MP3 Grounded Q&A: working rules

- MP3 **extends** MP2 (`../AI-RAG_MP2_MiniRAG/learner/mp2_rag.py`, v4). Carry MP2's pipeline forward. Never edit MP2.
- All new code goes in `learner/`. The plan is in `PLAN.md`.
- **The API keys in `learner/.env` are live and cost-sensitive (Vocareum proxy, about $1).**
  - No OpenAI or Qdrant call without the user's explicit go-ahead for that step.
  - Prefer offline checks (`selftest`, the cache, `sweep`). Never print or commit `.env`.
- The openai SDK needs an explicit `base_url` from `OPENAI_API_BASE` (Vocareum). Load `.env` via `python-dotenv`.
- Windows/PowerShell: force UTF-8 stdout, and write eval output with `--out` (UTF-8), not `>`.
- Constraints from the brief: `gpt-4o-mini` + `text-embedding-3-small` + Qdrant only, no LangChain/LlamaIndex, no deps beyond MP2's.
