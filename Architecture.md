# Architecture

> This file is maintained by Copilot and rewritten on every implementation step. It always reflects the current state of the project.

## Overview

**LLM-for-all** is a course-driven project building up an LLM-based application incrementally, lesson by lesson. It uses LangChain with OpenRouter to interact with LLMs. The workspace root **is** the project — all lessons add files directly here, blending into a single unified structure.

## Package Manager

- **uv** (by Astral) — replaces pip, venv, pyenv, and poetry
- Installed via `winget install --id=astral-sh.uv -e`
- All dependencies managed through `pyproject.toml` + `uv.lock` at the workspace root
- No bare `pip install` used anywhere in this project

## Project Structure

```
LLM-for-all/                         ← workspace root = project root
├── .github/
│   └── copilot-instructions.md      # Copilot session rules
├── .venv/                           # Virtual environment (gitignored)
├── .env                             # API key — NOT committed to git
├── .gitignore
├── .python-version                  # Python 3.12
├── architecture.md                  # This file
├── pyproject.toml                   # Project metadata and dependencies
├── uv.lock                          # Deterministic lockfile — committed to git
├── README.md
├── 01_human_message.py              # First LLM request script
└── lessons/                         # gitignored — lesson text + per-lesson resource diffs
    ├── 01-installing-uv/
    │   └── installing-uv.md         # No resources (lesson introduced no project files)
    └── 02-openrouter-first-request/
        ├── openrouter-first-request.md
        └── resources/               # Only files added/changed in lesson 2
            ├── .python-version
            ├── pyproject.toml
            ├── uv.lock
            └── 01_human_message.py
```

Each `resources/` folder contains **only the files introduced or modified in that lesson**, mirroring the live project root path. Applying lesson resources in order reproduces the full project state at any point.

## Dependencies

| Package            | Version | Role                                                |
| ------------------ | ------- | --------------------------------------------------- |
| `langchain`        | ≥ 1.3.0 | Core framework                                      |
| `langchain-openai` | ≥ 1.2.1 | OpenAI-compatible API client (used with OpenRouter) |
| `python-dotenv`    | ≥ 1.2.2 | Load secrets from `.env`                            |

## LLM Access

- **OpenRouter** — unified gateway to GPT, Claude, Gemini, Llama, and others
- Compatible with the OpenAI API; uses `ChatOpenAI` with `base_url="https://openrouter.ai/api/v1"`
- API key stored in `.env` as `OPENROUTER_API_KEY`

## Lessons Completed

| Folder                         | Topic                                                                   |
| ------------------------------ | ----------------------------------------------------------------------- |
| `01-installing-uv/`            | uv package manager — installation, commands, lockfiles                  |
| `02-openrouter-first-request/` | OpenRouter setup, first `HumanMessage` request, secure API key handling |

## Next

Transformers, self-attention, and prompt engineering — how GPT works under the hood.
