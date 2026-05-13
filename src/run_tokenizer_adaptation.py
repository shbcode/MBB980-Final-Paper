"""CLI: python -m src.run_tokenizer_adaptation --config configs/adaptation.yaml"""

from __future__ import annotations

import logging
import sys
import traceback

import click

from .experiments.tokenizer_adaptation import run
from .utils import load_config, new_run


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--log-level", default="INFO")
def main(config_path: str, log_level: str) -> None:
    cfg = load_config(config_path)
    run_ctx = new_run(cfg.get("output_dir", "results/tokenizer_adaptation"),
                      config=cfg, name="adaptation", log_level=log_level)
    try:
        metrics = run(dict(cfg))
        run_ctx.finalize(metrics=metrics)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        logging.getLogger(__name__).error("Tokenizer-adaptation experiment failed:\n%s", tb)
        run_ctx.add_extra("error", str(e))
        run_ctx.add_extra("traceback", tb)
        run_ctx.finalize()
        click.echo(f"Tokenizer-adaptation experiment failed: {e}\n\nFull traceback in "
                   f"{run_ctx.runs_dir}/run.log", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
