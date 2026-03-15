# gradtop

A terminal UI for monitoring neural network training in real time. Inspired by htop.

## What it shows

- Loss curve over training steps
- Per-layer update/data ratio (log10 scale) — the ratio of gradient updates to weight magnitudes, following Karpathy's diagnostic from the Neural Networks: Zero to Hero series. Values should hover around -3. Above -2 means large updates, below -4 means vanishing gradients.

## Installation

pip install gradtop

## Development

If you want to build from source, you need Rust and maturin:

```bash
git clone https://github.com/yourname/gradtop
cd gradtop
pip install maturin
maturin develop
```

## Usage

```python
import gradtop
from gradtop import gradtop

with GradTop(model, optimizer, every_n_steps=10) as monitor:
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            monitor.tick(loss.item())
```

Press any key to close the TUI. The terminal is restored automatically on exit.

