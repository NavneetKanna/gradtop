import gradtop
from collections import deque
import random
import time

class gradtopp:
    def __init__(self, model) -> None:
        monitor = gradtop.Monitor()

        loss = 1.0
        for step in range(200):
            if not monitor.is_running():
                break
            time.sleep(0.8)

            loss = loss * 0.98 + (random.random() * 0.05)

            monitor.tick(float(loss))

        self.model = model
        self.handles = []

    def _get_activation_hook(self, name):
        """Hook for Layers (ReLUs, Linear)"""
        def hook(module, grad_input, grad_output):
            g = grad_output[0]
            if g is None:
                return

            if isinstance(module, nn.ReLU):
                dead_count = (g <= 0).sum().item()
                total = g.numel()
                self.stats[f"{name}_dead_pct"] = (dead_count / total) * 100

        return hook

    def _get_weight_hook(self, name):
        """Hook for Parameters (Weights)"""
        def hook(grad):
            if grad is None: return

            g_norm = grad.norm().item()

            self.stats[f"{name}_norm"] = g_norm

        return hook

    def __enter__(self):
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                handle_module = module.register_full_backward_hook(self._get_activation_hook(name))
                self.handles.append(handle_module)

                for param_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        full_param_name = f"{name}.{param_name}"

                        handle_param = param.register_hook(self._get_weight_hook(full_param_name))
                        self.handles.append(handle_param)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()


gradtopp(None)
