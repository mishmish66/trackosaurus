"""Write pdoc pages of trex and every module into OUT (default site/).

    uv run python docs/build.py [OUT]
"""

import pkgutil
import sys
from pathlib import Path

import pdoc

import trex

modules = ["trex", *(f"trex.{m.name}" for m in pkgutil.iter_modules(trex.__path__) if m.name != "__main__")]
pdoc.pdoc(*modules, output_directory=Path(sys.argv[1] if len(sys.argv) > 1 else "site"))
