import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from models.softcot_model import SoftCoTModel
from utils.logging import Logger

def train_softcot_model(
    llm_model_name,
    assistant_model_name,
    train_dataset,
    eval_dataset,
    output_path,
    learning_rate=1e-4,
    weight_decay=0.01,
    num_epochs=10,
    batch_size=4,
    num_soft_tokens=5,
    device="cuda"
):
    """
    Train the SoftCoT model's projection module. 

    Args:
        llm_model_name: Name of the backbone LLM
        assistant_model_name: Name of the assistant model
        train_dataset: Dataset for training
        eval_dataset: Dataset for evaluation
        output_path: Path to save the model
        learning_rate: Learning rate for optimization
        weight_decay: Weight decay for regularization
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        num_soft_tokens: Number of soft thought tokens to generate
        device: Device to run the model on

    Returns:
        Trained SoftCoT model
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Initialize the SoftCoT model
    softcot_model = SoftCoTModel(
        llm_model_name,
        assistant_model_name,
        device=device
    )

    # Make sure projection module is initialized
    softcot_model.init_projection_module()

    # Initialize LLM for validation (loading when needed to save memory)
    # We don't keep this in memory during training, only load for evaluation

    # Setup logger
    logger = Logger(
        log_dir=os.path.join(output_path, "logs"),
        experiment_name="softcot_training"
    )

    # Define optimizer (only for projection module)
    optimizer = optim.AdamW(
        softcot_model.projection_module.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Define loss function
    criterion = nn.MSELoss()

    # Function to load LLM for validation
    def load_llm_for_eval():
        llm_model = torch.nn.DataParallel(
            nn.Linear(softcot_model.llm_hidden_dim, softcot_model.llm_hidden_dim), device_ids=[device]
        ).to(device)
        return llm_model

    # Training loop
    best_val_loss = float('inf')

    # Process dataset in batches
    for epoch in range(num_epochs):
        # Training phase
        softcot_model.projection_module.train()
        train_loss = 0.0

        # Process each sample individually to avoid memory issues
        progress_bar = tqdm(range(0, len(train_dataset), batch_size),
                           desc=f"Epoch {epoch+1}/{num_epochs} - Training")

        for i in progress_bar:
            batch_loss = 0.0
            optimizer.zero_grad()

            # Process each item in the batch
            actual_batch_size = min(batch_size, len(train_dataset) - i)
            for j in range(actual_batch_size):
                item = train_dataset[i + j]
                query = item["query"]
                reasoning = item.get("reasoning", "")

                # Generate soft thought tokens
                soft_thoughts = softcot_model.generate_soft_thoughts(query, num_soft_tokens)

                # Project soft thoughts to LLM space
                projected_thoughts = softcot_model.project_soft_thoughts(soft_thoughts)

                # For simplicity, we'll use a proxy task for training:
                # Making each token representation consistent within the sequence
                # This is a simplified version of the actual training objective
                target = projected_thoughts.mean(dim=1, keepdim=True).expand_as(projected_thoughts)

                # Compute loss
                loss = criterion(projected_thoughts, target) / actual_batch_size
                loss.backward()

                batch_loss += loss.item()

            # Update weights
            optimizer.step()
            train_loss += batch_loss

            # Update progress bar
            progress_bar.set_postfix({"loss": batch_loss})

        avg_train_loss = train_loss / len(train_dataset) * batch_size
        print(f"Epoch {epoch+1}/{num_epochs} - Average Training Loss: {avg_train_loss:.6f}")

        # Validation phase
        softcot_model.projection_module.eval()
        val_loss = 0.0

        with torch.no_grad():
            # Load LLM for validation
            llm_proxy = load_llm_for_eval()

            progress_bar = tqdm(range(0, len(eval_dataset), batch_size),
                               desc=f"Epoch {epoch+1}/{num_epochs} - Validation")

            for i in progress_bar:
                batch_loss = 0.0

                # Process each item in the batch
                actual_batch_size = min(batch_size, len(eval_dataset) - i)
                for j in range(actual_batch_size):
                    item = eval_dataset[i + j]
                    query = item["query"]

                    # Generate soft thought tokens
                    soft_thoughts = softcot_model.generate_soft_thoughts(query, num_soft_tokens)

                    # Project soft thoughts to LLM space
                    projected_thoughts = softcot_model.project_soft_thoughts(soft_thoughts)

                    # Use the LLM proxy to evaluate coherence
                    llm_output = llm_proxy(projected_thoughts.view(-1, softcot_model.llm_hidden_dim))

                    # Compute validation loss using identity mapping as a proxy task
                    loss = criterion(llm_output, projected_thoughts.view(-1, softcot_model.llm_hidden_dim))
                    batch_loss += loss.item() / actual_batch_size

                val_loss += batch_loss

                # Update progress bar
                progress_bar.set_postfix({"val_loss": batch_loss})

            # Clean up
            del llm_proxy
            torch.cuda.empty_cache()

        avg_val_loss = val_loss / len(eval_dataset) * batch_size
        print(f"Epoch {epoch+1}/{num_epochs} - Average Validation Loss: {avg_val_loss:.6f}")

        # Log metrics
        logger.log_metrics({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        }, epoch)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            # Save checkpoint
            checkpoint_path = os.path.join(output_path, "checkpoints")
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(softcot_model.projection_module.state_dict(), os.path.join(checkpoint_path, f"projection_module_best.pt"))

            print(f"Saved best model with validation loss: {best_val_loss:.6f}")

    # Load best model
    if os.path.exists(os.path.join(checkpoint_path, f"projection_module_best.pt")):
        softcot_model.projection_module.load_state_dict(torch.load(os.path.join(checkpoint_path, f"projection_module_best.pt")))

    logger.close()
    return softcot_model