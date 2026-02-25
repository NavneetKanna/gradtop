import gradtop
from collections import deque
import random
import time
from torch import nn
import sys

class gradtopp:
    def __init__(self, model, every_n_steps: int = 10) -> None:
        self.monitor = gradtop.Monitor()

        self.model = model
        self.handles = []

        self._grad_norms: dict[str, float] = {}
        self._weight_norms: dict[str, float] = {}

        self._step = 0
        self.every_n_steps = every_n_steps

    def tick(self, loss: float):
        self._step += 1
        if self._step % self.every_n_steps != 0:
            return

        names = list(self._grad_norms.keys())

        self.monitor.tick(
            loss,
            names,
            [self._grad_norms[n] for n in names],
            [self._weight_norms[n] for n in names],
        )

        self._grad_norms.clear()
        self._weight_norms.clear()

    def _get_activation_hook(self, name):
        def hook(module, grad_input, grad_output):
            pass

        return hook

    def _get_weight_hook(self, name, param):
        def hook(grad):
            if grad is None:
                return
            self._grad_norms[name] = grad.norm().item()
            self._weight_norms[name] = param.norm().item()

        return hook

    def __enter__(self):
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                handle_module = module.register_backward_hook(self._get_activation_hook(name))
                self.handles.append(handle_module)

                for param_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        full_param_name = f"{name}.{param_name}"

                        handle_param = param.register_hook(self._get_weight_hook(full_param_name, param))
                        self.handles.append(handle_param)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()

