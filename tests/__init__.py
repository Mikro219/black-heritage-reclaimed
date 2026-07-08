"""BHR regression test suite.

Run after ANY engine/detector change:

    py -3.12 -m unittest discover -s tests -v

Covers: detector behaviour (playtest contract), depth fusion, tutorial flow,
render engine (headless), and the shot sequence / metadata invariants.
No camera, display, or Gemini hardware required.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
