# Import classes from internal modules
from .board import Board
from .bot import Bot
from .button import Button
from .gui import Gui
from .model import ChessNet
from .piece import Piece
from .startmenu import StartMenu
from .instructionpopup import InstructionPopup
from pathlib import Path
import pandas as pd

# Import utils
import utils as ut

# Import constants from config
from config import HEIGHT, WIDTH, FPS, DIMENSION, CHESS_NOTATION_TO_IMAGE_NAMES, SQ_SIZE, ROOT

# Load once at package import
BASE_DIR = Path(__file__).parent.parent
uci_indexes = pd.read_csv(BASE_DIR / 'uci_index.csv')
uci_to_index = dict(zip(uci_indexes['uci'], uci_indexes['index']))
index_to_uci = {v: k for k, v in uci_to_index.items()}
uci_index_connector = {**uci_to_index, **index_to_uci} # uci_to_index_connector is a dictionary that can be used in two ways: retrieve the move index with a uci, or the uci with a move index.

# Define what is exposed when importing *
__all__ = [
    "Board",
    "Bot",
    "Button",
    "Gui",
    "ChessNet",
    "Piece",
    "StartMenu",
    "InstructionPopup"
    "ut",
    "HEIGHT",
    "WIDTH",
    "FPS",
    "DIMENSION",
    "CHESS_NOTATION_TO_IMAGE_NAMES",
    "SQ_SIZE",
    "ROOT",
    'uci_index_connector'
]