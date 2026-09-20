"""Experiment modules. Each exposes:

    cells(seeds) -> list[dict]       environment-side cells (regime, seed, ...)
    scenario(cell, ctx) -> Scenario  the world plus its drift/notice/probe hooks
    REFERENCE_AGENTS: list[dict]     default in-process agents (plugins; optional)
    analyze(run_dir)                 tables from logs, agent-agnostic
"""
