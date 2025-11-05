import sys
from pathlib import Path

# Add src/ to Python module search path
SRC_DIR = Path(__file__).parent / "src"
sys.path.append(str(SRC_DIR))

# import the chessgui package
from chessgui import Gui

if __name__ == "__main__":

    gui = Gui()
    gui.run()