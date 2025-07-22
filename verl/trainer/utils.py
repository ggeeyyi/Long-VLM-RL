import torch
import numpy as np
from tensordict import TensorDict

def convert_rollout_batch_to_tensordict(rollout_batch):
    """
    Convert rollout_batch from HuggingFace dataset format to TensorDict
    Handles both dictionary and numpy array inputs
    """
    tensor_dict_data = {}
    
    # Handle case where rollout_batch is a numpy array (from HuggingFace dataset)
    if isinstance(rollout_batch, np.ndarray):
        # If it's a numpy array, it likely contains structured data
        # Try to extract the dictionary from the first element if it's an object array
        if rollout_batch.dtype == object and len(rollout_batch) > 0:
            # Take the first element which should be the dictionary
            rollout_batch = rollout_batch.item() if rollout_batch.ndim == 0 else rollout_batch[0]
        else:
            raise ValueError(f"Cannot convert numpy array of dtype {rollout_batch.dtype} to TensorDict. Expected object array containing dictionaries.")
    
    # Handle case where rollout_batch is already a dictionary
    if hasattr(rollout_batch, 'items'):
        for key, value in rollout_batch.items():
            if isinstance(value, list):
                # Convert nested lists to tensors
                tensor_dict_data[key] = torch.tensor(value)
            elif isinstance(value, np.ndarray):
                # Convert numpy arrays to tensors
                tensor_dict_data[key] = torch.from_numpy(value)
            elif isinstance(value, torch.Tensor):
                tensor_dict_data[key] = value
            else:
                # For other types, try to convert to tensor
                tensor_dict_data[key] = torch.tensor([value]) if not isinstance(value, torch.Tensor) else value
    else:
        raise ValueError(f"rollout_batch must be a dictionary or numpy array, got {type(rollout_batch)}")
    
    return TensorDict(tensor_dict_data)