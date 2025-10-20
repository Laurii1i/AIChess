import pandas as pd

uci_indexes = pd.read_csv(r"C:\Users\lauri\Documents\Chess\AIChess\uci_index.csv")
uci_to_index = dict(zip(uci_indexes['uci'], uci_indexes['index']))
index_to_uci = {v: k for k, v in uci_to_index.items()}
two_way = {**uci_to_index, **index_to_uci}