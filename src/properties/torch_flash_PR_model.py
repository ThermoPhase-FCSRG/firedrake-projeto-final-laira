import torch_flash

print([
    name for name in dir(torch_flash)
    if "peng" in name.lower() or "pr" in name.lower()
])