# Chunk: docs/chunks/stream_visualization - CLI entry point for visualization
"""Entry point for ``python -m repro_maa.visualize``.

This thin wrapper delegates to :func:`repro_maa.visualize.cli_main`.
"""
from repro_maa.visualize import cli_main

cli_main()
