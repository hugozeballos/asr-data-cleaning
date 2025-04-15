from accelerate import Accelerator
import torch

accelerator = Accelerator()
device = accelerator.device
print(f"Accelerate está usando el dispositivo: {device}")

if torch.cuda.is_available():
    print(f"Total de GPUs detectadas: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
