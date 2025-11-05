# AIChess: Deep Learning Chess AI

## Overview

AIChess is a chess engine trained using deep neural networks to evaluate positions and predict optimal moves. The network uses residual blocks to capture hierarchical features from board positions, supporting both policy prediction (best move) and value evaluation (position strength).

## Features

* The package contains a small example dataset of 10 000 chess positions, which is used to demonstrate the training process. This data is located at /src/training/data/example_training_data
* The neural network (NN) was trained on ~ 50 000 000 chess positions with associated best moves, position evaluations. The data was obtained from Kaggle, URL: https://www.kaggle.com/datasets/lichess/chess-evaluations.
* Chess positions (FENs) from the data were encoded into tensor represenation for the NN.
* Best moves and centipawn scores were extracted from the data for computing the loss.
* The NN was trained to predict the policy head and the value head.
    * Policy head predicts the best move for a given board position.
    * Value head evaluates the position on a scale from -1 (black advantage) to 1 (white advantage). Analogous to the centipawn scores.
    * I didn’t use a separate validation dataset because the Kaggle dataset was already highly diverse and contained far more analyzed chess positions than I could feasibly train on. (I know this is bad practice)
    * The source code for the developed model is located at /src/model/model.py
* Research was conducted for the neural network architecture. The choice for the architecture is explained under section ## Architecture
* A GUI was built using the Pygame module to provide an interactive interface for playing against the NN. Source for this code is located at /src/chessgui/

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Laurii1i/AIChess.git
```

2. Create a virtual environment (recommended):

```bash
cd AIChess
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies, exluding PyTorch:

```bash
pip install -r requirements.txt
```

4. Installing PyTorch

Install the CPU version (simple installation, slow performance in training):
```bash
pip install torch torchvision
```
Optionally, you can install the GPU-enabled version of PyTorch from the following link:
🔗 https://pytorch.org/get-started/locally/


## Training the model

> **Note:** The following imports assume you are running the script from the `root/src/` directory. Alternatively, you can run train.py for a training demonstration and main.py for playing against the bot.

```python
from model import ChessNet
from training import Trainer

# Initialize the model
model = ChessNet()
trainer = Trainer(model)

# Train the model
trainer.train()
```

## Playing against the NN. 

```python
from chessgui import Gui

gui = Gui()
gui.run()

```
## Dataset

* Training data consists of Parquet files containing chess positions, best moves, and evaluation scores.
* A small example dataset is included: `example_training_data.parquet`, which is used for the training demonstration.

## Contributing

Contributions are welcome! Please submit pull requests or issues for bug fixes, enhancements, or ideas.

## License

This project is licensed under the MIT License.

## Architecture

### Input Tensor Encoding

Each board position is encoded as a 13×8×8 tensor:

12 planes for pieces (6 for white, 6 for black) representing their presence on each square.

1 additional plane for side-to-move (1 for white, 0 for black).

This encoding preserves spatial information of the board and allows the convolutional layers to detect local patterns (e.g., pawn structures, attacks, defenses) and propagate them to deeper layers for high-level positional understanding.

### Output Move Space

The policy head outputs a vector of size 1968, representing all possible legal moves in UCI format mapped to indices. This uci-to-index-mapping is found in csv-format at /src/uci_index.csv"

To handle illegal moves: a mask vector of length 1968 is generated for each position, with True for legal moves and False for illegal moves. This ensures the network only assigns probability to valid moves.

The value head outputs a single scalar in the range [-1, 1], -1 meaning that black is dominating the position and likewise +1 for white. This encoding works because of the following factors: 

* Spatial preservation: Convolutional layers can exploit patterns like piece arrangements, pawn chains, and king safety.

* Large action space coverage: Every possible legal move is represented and masked, enabling the network to learn a complete policy.

* Compatibility with residual blocks: Hierarchical features can be refined across deep residual layers without losing positional context.

This design ensures the network can simultaneously learn where to move and how good the position is.

### Defining the loss

The training data contained board position specific centipawn scores and stockfish-evaluated best moves, which were used to compute the loss during the training. An equal contribution was adapted from both the policy network and the value network. This was done because I estimated that it is "as important to understand how good a position is, as to find the best move". 

### Neural network

In designing a neural network to train a chess AI — one that learns policy and value for positions (e.g., board state → move distribution + evaluation) — the architecture must satisfy several requirements:

* It must be able to extract spatial features from board representations (e.g., a 13×8×8 tensor) and propagate them through many layers so that both low‑ and high‑level      positional patterns can be learned.

* It must maintain stable gradient flows even when many layers are stacked, so training remains efficient and doesn’t degrade with depth.

* It must support two distinct heads: (1) a policy head that outputs a large action‑space (1968 moves) and (2) a value head producing a scalar evaluation in [–1, 1]. This sort of two-headed approach was motivated by Alphazero (the strongest chess engine in the world), which was trained using this strategy.

* It must generalize well — i.e., not overfit to known positions but be able to respond to novel ones.

Given these demands, the architecture I chose consists of:

* A “stem” convolutional block to map input planes → feature filters.

* A residual‑tower of many residual blocks (in our case ~10 such blocks) to extract hierarchical features while preserving gradient flow.

* Two separate heads: a policy head and a value head with final tanh activation to confine value between [-1, 1].

* Weight initialization (Kaiming) and normalization (BatchNorm) to further stabilize training.

### Why residual blocks (skip‑connection blocks) are a particularly good architectural choice for chess AI domain?

Residual networks (ResNets) introduced by He et al. (2015) revolutionized deep network architectures by enabling very deep networks via skip connections or identity mappings within residual blocks [He et al., “Deep Residual Learning for Image Recognition”, 2015]. The key idea: each block learns a mapping 𝐹(𝑥)+𝑥 instead of a full mapping 𝐻(𝑥). This seemingly small change has profound effects:

The skip connection allows the gradient (and the signal) to flow more directly backward (and forward) through the network. This prevents earlier layers from “dying out” because they receive no gradient signal. As a result, networks of many dozens or even hundreds of layers can be trained more reliably. Furthermore, Zaeemzadeh et al. show theoretically that residual blocks preserve the norm of the gradient and enable stable back‑propagation when many blocks are stacked. 
(Ref: https://arxiv.org/abs/1805.07477)

In a deep network, not every layer must learn a major transformation — sometimes the optimal mapping is “just pass the signal through”. In a plain feed‑forward network this can be hard to learn (it would require weights to exactly copy). In a residual block, the identity mapping is the default (via the skip), and the block only needs to learn what to add. This simplifies optimization. Deeper networks can avoid degradation because residual blocks allow them to behave like shallower networks if needed (i.e., if additional layers are not helpful). 
(Ref: https://yashwantherukulla.github.io/Papers/ResNet---Deep-Residual-Learning-for-Image-Recognition)

Inductive bias suitable for hierarchical spatial domains
Chess evaluation is inherently hierarchical: local patterns (pawn structure, king safety) → mid‑game structure → end‑game patterns. A deep residual network is well‑suited to iteratively refine representations: early blocks extract low‑level spatial features; deeper blocks adjust and combine them. Because residual blocks allow “incremental refinement” rather than forcing each block to relearn full transforms, the architecture suits domains like chess where refinement and feature reuse matter. Indeed, papers like “Why ResNet Works? Residuals Generalize” show that residual connections do not increase hypothesis complexity (i.e., they don’t necessarily over‑fit more) but improve generalization. 
(Ref: https://www.emergentmind.com/papers/1904.01367)

Architectural Choices Summary & Alignment with Chess Domain

* Stem convolution (3×3 kernel, padding=1) ensures the entire 8×8 board is treated with full receptive field from the outset.

* Residual tower with repeated blocks: allows deep refinement of features while keeping training stable.

Policy head: reduces feature channels via 1×1 conv to 32, flattens (32×8×8) → linear to large output size (action_size=1968) matching move vocabulary.

Value head: similar structure but compresses to scalar with tanh activation (to map to [–1,1], matching evaluation scaling).

Normalization (BatchNorm): improves training stability; papers show batch normalization in residual networks biases blocks toward identity at initialization, promoting learning stability. 
(Ref: https://arxiv.org/abs/2002.10444)

Weight initialization (Kaiming): tailored for ReLU activations and deep architectures.


