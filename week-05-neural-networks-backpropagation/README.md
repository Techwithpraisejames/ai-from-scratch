# How Neural Networks Learn: Backpropagation Explained with Python

**AI From Scratch — Week 5**

In this lesson, you'll train a small neural network on the XOR problem and observe how its predictions and parameters change during training. This follows [Week 4's prompt engineering experiment](../week-04-prompt-engineering/README.md), where we tested instructions with an API without training the model.

## What you'll learn

- **Neurons and parameters:** A neuron combines inputs with learned weights and a bias.
- **Nonlinear activation:** A hidden `Tanh` layer allows the network to model a pattern that a single linear decision boundary cannot represent.
- **Forward propagation:** The model computes output scores from the inputs.
- **Loss:** `BCEWithLogitsLoss` measures error against the binary targets, using logits for numerical stability.
- **Backpropagation:** `loss.backward()` uses automatic differentiation and the chain rule to compute parameter gradients.
- **Optimization:** `optimizer.step()` uses those gradients to update parameters. Backpropagation calculates gradients; the optimizer performs the updates.

## The XOR problem

| Input A | Input B | Target |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

The model architecture is **2 inputs → 4 hidden neurons with Tanh → 1 output logit**. Sigmoid converts the output logit into a probability for display. The model trains on all four XOR examples, so its final performance demonstrates fitting this tiny dataset, **not generalization to unseen data**.

## Files

- [`train_xor.py`](train_xor.py): complete training experiment
- [`requirements.txt`](requirements.txt): dependency range
- [`exercises.md`](exercises.md): experiments to try after the first run

## Run locally

Use Python **3.10–3.13** (64-bit) and VS Code. PyTorch installation can vary by operating system and hardware; see [PyTorch's official installation instructions](https://pytorch.org/get-started/locally/) if pip cannot find a compatible wheel.

### Windows PowerShell

From the repository root, enter this lesson folder:

```powershell
cd week-05-neural-networks-backpropagation
```

If you already have a working virtual environment in the repository root from Week 3, you can reuse it:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe train_xor.py
```

Otherwise, create a new environment inside this folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_xor.py
```

If you have a different supported Python version, change `-3.13` accordingly.

### macOS / Linux

```bash
cd week-04-neural-networks-backpropagation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_xor.py
```

## What to look for

The terminal shows initial predictions, a sample weight before training, loss at epochs 1 and 500–3000, the same weight after training, and final probabilities and predicted classes.

With the supplied seed and settings, the predicted classes should usually become `[0, 1, 1, 0]`. Exact losses, weights, and probabilities can differ across PyTorch versions and platforms. If they don't converge, preserve your terminal output and check the environment before modifying the experiment.

The training cycle is:

```text
inputs → model(inputs) → loss → loss.backward() → optimizer.step()
```

`optimizer.zero_grad()` clears gradients accumulated from the previous step. `torch.no_grad()` disables gradient tracking when inspecting predictions.

## Reproducibility

The example sets random seeds and uses CPU-compatible operations. The dependency file specifies a supported PyTorch range rather than a single platform-specific build; record `python --version` and `python -m pip show torch` if you need to reproduce an exact run. The network is a teaching example, not a production classifier.

Complete the [exercises](exercises.md) after your first run.
