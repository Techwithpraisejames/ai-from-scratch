"""Train a tiny neural network on XOR and inspect its learning process."""

import random

import torch
from torch import nn


SEED = 42
EPOCHS = 3000
LEARNING_RATE = 0.1


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    inputs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    targets = torch.tensor([[0.], [1.], [1.], [0.]])

    model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def predictions():
        with torch.no_grad():
            return torch.sigmoid(model(inputs)).flatten().tolist()

    print("XOR: inputs [0,0], [0,1], [1,0], [1,1]")
    print("Targets: [0, 1, 1, 0]")
    print("Initial probabilities:", [round(p, 3) for p in predictions()])
    first_weight = model[0].weight[0, 0].item()
    print(f"Initial example weight: {first_weight:.6f}")

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        logits = model(inputs)             # forward propagation
        loss = criterion(logits, targets)  # measure error
        loss.backward()                    # backpropagation: calculate gradients
        optimizer.step()                   # use gradients to update parameters

        if epoch == 1 or epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | loss: {loss.item():.6f}")

    print(f"Final example weight: {model[0].weight[0, 0].item():.6f}")
    final = predictions()
    print("Final probabilities:", [round(p, 3) for p in final])
    print("Predicted classes:", [int(p >= 0.5) for p in final])
    print("Expected classes: [0, 1, 1, 0]")


if __name__ == "__main__":
    main()
