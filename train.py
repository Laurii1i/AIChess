import sys
from pathlib import Path

# Add src/ to Python module search path
SRC_DIR = Path(__file__).parent / "src"
sys.path.append(str(SRC_DIR))

from model import ChessNet
from training import Trainer


if __name__ == "__main__":

    batch_size = 32
    model = ChessNet()
    trainer = Trainer(model, batch_size=batch_size)

    trainer.train()        

