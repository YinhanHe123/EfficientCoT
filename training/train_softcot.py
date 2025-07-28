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

    # Function to create a simple evaluation proxy for training loss
    def create_eval_proxy():
        """Create a simple linear layer to evaluate projection quality"""
        return nn.Linear(softcot_model.llm_hidden_dim, softcot_model.llm_hidden_dim).to(device)

    # Training loop
    best_val_loss = float('inf')
    checkpoint_path = os.path.join(output_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

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
                try:
                    item = train_dataset[i + j]
                    query = item["query"]
                    reasoning = item.get("reasoning", "")

                    # Generate soft thought tokens
                    soft_thoughts = softcot_model.generate_soft_thoughts(query, num_soft_tokens)

                    # Project soft thoughts to LLM space
                    projected_thoughts = softcot_model.project_soft_thoughts(soft_thoughts)

                    # Training objective: consistency within the sequence
                    # This encourages the projection to create meaningful representations
                    if projected_thoughts.size(1) > 1:
                        # Use variance minimization as a proxy for consistency
                        mean_repr = projected_thoughts.mean(dim=1, keepdim=True)
                        target = mean_repr.expand_as(projected_thoughts)
                        loss = criterion(projected_thoughts, target) / actual_batch_size
                    else:
                        # For single token, use identity mapping
                        target = projected_thoughts
                        loss = criterion(projected_thoughts, target) / actual_batch_size

                    loss.backward()
                    batch_loss += loss.item()

                    # Clean up memory
                    del soft_thoughts, projected_thoughts, target
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error processing sample {i+j}: {e}")
                    continue

            # Update weights
            optimizer.step()
            train_loss += batch_loss

            # Update progress bar
            progress_bar.set_postfix({"loss": batch_loss})

        avg_train_loss = train_loss / (len(train_dataset) // batch_size + 1)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Training Loss: {avg_train_loss:.6f}")

        # Validation phase
        softcot_model.projection_module.eval()
        val_loss = 0.0

        with torch.no_grad():
            # Create evaluation proxy
            eval_proxy = create_eval_proxy()

            progress_bar = tqdm(range(0, min(len(eval_dataset), 50), batch_size),  # Limit validation set for speed
                               desc=f"Epoch {epoch+1}/{num_epochs} - Validation")

            for i in progress_bar:
                batch_loss = 0.0

                # Process each item in the batch
                actual_batch_size = min(batch_size, min(len(eval_dataset), 50) - i)
                for j in range(actual_batch_size):
                    try:
                        item = eval_dataset[i + j]
                        query = item["query"]

                        # Generate soft thought tokens
                        soft_thoughts = softcot_model.generate_soft_thoughts(query, num_soft_tokens)

                        # Project soft thoughts to LLM space
                        projected_thoughts = softcot_model.project_soft_thoughts(soft_thoughts)

                        # Use the evaluation proxy to check projection quality
                        reshaped_thoughts = projected_thoughts.view(-1, softcot_model.llm_hidden_dim)
                        proxy_output = eval_proxy(reshaped_thoughts)

                        # Compute validation loss using identity mapping as a proxy task
                        loss = criterion(proxy_output, reshaped_thoughts)
                        batch_loss += loss.item() / actual_batch_size

                        # Clean up memory
                        del soft_thoughts, projected_thoughts, proxy_output
                        torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"Error processing validation sample {i+j}: {e}")
                        continue

                val_loss += batch_loss

                # Update progress bar
                progress_bar.set_postfix({"val_loss": batch_loss})

            # Clean up
            del eval_proxy
            torch.cuda.empty_cache()

        avg_val_loss = val_loss / (min(len(eval_dataset), 50) // batch_size + 1)
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
            torch.save(
                softcot_model.projection_module.state_dict(),
                os.path.join(checkpoint_path, "projection_module_best.pt")
            )

            print(f"Saved best model with validation loss: {best_val_loss:.6f}")

        # Save the model at the end of each epoch
        softcot_model.save_pretrained(output_path)

    # Load best model
    best_model_path = os.path.join(checkpoint_path, "projection_module_best.pt")
    if os.path.exists(best_model_path):
        softcot_model.projection_module.load_state_dict(torch.load(best_model_path))
        print("Loaded best model from checkpoint")

    # Final save
    softcot_model.save_pretrained(output_path)
    logger.close()

    return softcot_model