import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
# print(torch.cuda.max_memory_reserved(0))
# # print(torch.cuda.max_memory_cached(0))
# print(torch.cuda.max_memory_allocated(0))
print(torch.version.cuda)

# torch.cuda.empty_cache()
# torch.cuda.ipc_collect()