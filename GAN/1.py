import torch
print("GPU可用:", torch.cuda.is_available())
print("GPU数量:", torch.cuda.device_count())
print("当前GPU:", torch.cuda.get_device_name(0))
print("显存使用情况:", torch.cuda.memory_allocated(0), "bytes")