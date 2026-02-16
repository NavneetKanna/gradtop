class gradtop:
    def __init__(self, model) -> None:
        self.model = model
        self.handles = []

    def hook_fn(self, module, grad_input, grad_output):
        print(f"Layer: {module.__class__.__name__} | Grad Output: {grad_output[0].shape if grad_output[0] is not None else None}")

    def __enter__(self):
        for name, module in self.model.named_modules():
            # print(f"Name: {name}, Type: {type(module)}")
            handle = module.register_full_backward_hook(self.hook_fn)
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()

