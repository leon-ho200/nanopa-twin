from __future__ import annotations

import logging
from typing import Any, cast

import tyro

from nanopa_twin.dial.commands import Encode, Evaluate, Export, Fit, Forecast

Command = Fit | Evaluate | Forecast | Encode | Export


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    command = tyro.cli(cast(Any, Command))
    command.run()
