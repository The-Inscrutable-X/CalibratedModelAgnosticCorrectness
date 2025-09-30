import gc
import torch
def get_tensor_info():
        tensor_info = []
        total_mem = {"cuda": 0, "cpu": 0}
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    if obj.is_cuda:
                        tensor_info.append({
                            'type': type(obj),
                            'size': obj.size(),
                            'memoryMB': obj.element_size() * obj.nelement() / 1024**2,  # Memory in MB
                            'device': obj.device
                        })
                        total_mem["cuda"] += obj.element_size() * obj.nelement() / 1024**3
                    if obj.device == "cpu":
                        total_mem["cpu"] += obj.element_size() * obj.nelement() / 1024**3
            except Exception:
                pass
        print("Total Memory in GB:", total_mem)
        return tensor_info

# Monitor GPU Usage
def monitor_all_gpus(logger=None):
    num_gpus = torch.cuda.device_count()
    for gpu_id in range(num_gpus):
        torch.cuda.set_device(gpu_id)  # Set the current GPU
        allocated_memory = torch.cuda.memory_allocated(gpu_id)
        reserved_memory = torch.cuda.memory_reserved(gpu_id)
        if logger:
            logger.info((f"GPU {gpu_id}:"
                        f"  Memory Allocated: {allocated_memory / 1024**2:.2f} MB"
                        f"  Memory Reserved: {reserved_memory / 1024**2:.2f} MB\n"))
        else:
            print((f"GPU {gpu_id}:"
                        f"  Memory Allocated: {allocated_memory / 1024**2:.2f} MB"
                        f"  Memory Reserved: {reserved_memory / 1024**2:.2f} MB\n"))
