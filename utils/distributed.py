import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    """
    Setup distributed training environment

    Args:
        rank: Rank of the current process
        world_size: Total number of processes
    """
    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set device for this process
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training environment"""
    dist.destroy_process_group()

def convert_model_to_ddp(model, device):
    """
    Convert a model to DistributedDataParallel

    Args:
        model: PyTorch model
        device: Device to place the model on

    Returns:
        DDP model
    """
    # Move model to device
    model = model.to(device)

    # Wrap the model with DDP
    model = DDP(model, device_ids=[device])

    return model

def is_main_process(rank):
    """Check if this is the main process (rank 0)"""
    return rank == 0

def reduce_loss(tensor, world_size):
    """
    Reduce a loss tensor across all processes

    Args:
        tensor: Loss tensor
        world_size: Total number of processes

    Returns:
        Reduced tensor
    """
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size