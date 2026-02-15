class gradtop:
    def __init__(self) -> None:
        pass

    def __enter__(self, model):
        for name, module in model.named_modules():
            print(f"Name: {name}, Type: {type(module)}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

