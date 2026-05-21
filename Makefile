.PHONY: install demo run mcp clean help

help:
	@echo "  make install  → install dependencies via uv"
	@echo "  make demo     → run in demo mode (no Spotify needed)"
	@echo "  make run      → run with your real Spotify account"
	@echo "  make mcp      → run the MCP server (for Claude Desktop)"
	@echo "  make clean    → wipe local DB + vector store"

install:
	@command -v uv >/dev/null 2>&1 || { echo "Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	uv sync
	@test -f .env || cp .env.example .env
	@echo "✓ Setup complete. Edit .env to add your API keys."

demo:
	uv run python -m backend.app --demo --port 3000

run:
	uv run python -m backend.app --port 3000

mcp:
	uv run python -m backend.mcp_server

clean:
	rm -rf data/archive.db data/chroma
	@echo "✓ Local data cleared."
