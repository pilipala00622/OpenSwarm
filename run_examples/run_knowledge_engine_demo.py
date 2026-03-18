"""
Demo: local knowledge engine retrieval without external APIs.

Usage:
    python3 run_examples/run_knowledge_engine_demo.py
"""

import importlib.util
import contextlib
import io
import os
import sys


def _load_open_swarm_package():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    init_path = os.path.join(root_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        "open_swarm",
        init_path,
        submodule_search_locations=[root_dir],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["open_swarm"] = module
    assert spec.loader is not None
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
    return module


open_swarm = _load_open_swarm_package()

KnowledgeEngine = open_swarm.KnowledgeEngine
RetrieveContextTool = open_swarm.RetrieveContextTool


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    engine = KnowledgeEngine(root_dir=repo_root)
    tool = RetrieveContextTool(knowledge_engine=engine, root_dir=repo_root)

    queries = [
        "team cleanup and running team tasks",
        "task claim release pending recovery",
        "handoff cross session context",
    ]

    for query in queries:
        print("=" * 72)
        print(f"QUERY: {query}")
        print("=" * 72)
        result = tool.knowledge_engine.search(query=query, top_k=3, include_related=True, refresh_index=False)
        if not result:
            print("No hits")
            continue
        for idx, hit in enumerate(result, 1):
            print(f"{idx}. {hit['path']} (score={hit['score']})")
            print(f"   reasons: {', '.join(hit['reasons']) or 'n/a'}")
            print(f"   symbols: {', '.join(hit['symbols']) or 'n/a'}")
            print(f"   related: {', '.join(hit['related_paths']) or 'n/a'}")
            print("   snippet:")
            for line in str(hit["snippet"]).splitlines()[:10]:
                print(f"   {line}")
            print()


if __name__ == "__main__":
    main()
