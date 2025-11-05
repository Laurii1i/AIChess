# Standard library imports
from pathlib import Path

# Third-party imports
import pandas as pd

ROOT = Path(__file__).parent.parent # root folder

uci_indexes = pd.read_csv(ROOT / 'src' /'uci_index.csv')
uci_to_index = dict(zip(uci_indexes['uci'], uci_indexes['index']))
index_to_uci = {v: k for k, v in uci_to_index.items()}
uci_index_connector = {**uci_to_index, **index_to_uci} # uci_index_connector('e2e4') -> 193    uci_index_connector(193) -> 'e2e4' 