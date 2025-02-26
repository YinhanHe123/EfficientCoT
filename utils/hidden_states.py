import torch

def inject_hidden_states(model_outputs, custom_hidden_states, layer_idx):
    """
    Inject custom hidden states into a model's outputs

    Args:
        model_outputs: Original model outputs with hidden states
        custom_hidden_states: Custom hidden states to inject
        layer_idx: Index of the layer to replace

    Returns:
        Modified model outputs with injected hidden states
    """
    # Copy all hidden states
    all_hidden_states = list(model_outputs.hidden_states)

    # Replace the specified layer with custom hidden states
    all_hidden_states[layer_idx] = custom_hidden_states

    # Create a new tuple of hidden states
    new_hidden_states = tuple(all_hidden_states)

    # Modify the outputs object to use new hidden states
    # Note: This depends on the specific model implementation
    model_outputs.hidden_states = new_hidden_states

    return model_outputs

def extract_last_token_hidden_states(hidden_states, attention_mask=None):
    """
    Extract hidden states for the last token in each sequence

    Args:
        hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim)
        attention_mask: Optional attention mask of shape (batch_size, seq_len)

    Returns:
        Tensor of shape (batch_size, hidden_dim) with last token states
    """
    if attention_mask is not None:
        # Find the last non-padding token in each sequence
        last_token_indices = attention_mask.sum(dim=1) - 1

        # Ensure indices are valid
        last_token_indices = torch.clamp(last_token_indices, min=0)

        # Extract the hidden states for the last token in each sequence
        batch_size = hidden_states.size(0)
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        last_token_states = hidden_states[batch_indices, last_token_indices]
    else:
        # If no attention mask, use the last token for all sequences
        last_token_states = hidden_states[:, -1]

    return last_token_states

def interpolate_hidden_states(hidden_states, target_length):
    """
    Interpolate hidden states to a target sequence length

    Args:
        hidden_states: Tensor of shape (batch_size, src_len, hidden_dim)
        target_length: Target sequence length

    Returns:
        Tensor of shape (batch_size, target_length, hidden_dim)
    """
    batch_size, src_len, hidden_dim = hidden_states.size()

    # Reshape for interpolation
    hidden_states_reshaped = hidden_states.permute(0, 2, 1)  # (batch_size, hidden_dim, src_len)

    # Use linear interpolation to resize
    interpolated = torch.nn.functional.interpolate(
        hidden_states_reshaped,
        size=target_length,
        mode='linear',
        align_corners=False
    )

    # Reshape back
    interpolated = interpolated.permute(0, 2, 1)  # (batch_size, target_length, hidden_dim)

    return interpolated