#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.pipeline import RAGPipeline


def main() -> None:
    settings = Settings.from_env()
    rag = RAGPipeline(settings=settings)
    result = rag.build_index()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
