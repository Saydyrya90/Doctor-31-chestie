#!/bin/bash
set -e

export PATH="$HOME/.local/bin:$PATH"

echo "📚 Generating HTML documentation with pdoc..."
pdoc src -o docs

echo "✅ Documentation ready at: docs/index.html"
