import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
import argparse
import os
import time
import pdb

# Define the individual layers for our MLP
class MLPLayer1(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # Flatten the input if it's not already flattened
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return F.relu(self.fc(x))

class MLPLayer2(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# TwoLayerMLP model (without pipeline parallelism) for comparison
class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=5):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Flatten the input if it's not already flattened
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# Training function for pipelined model
def train_pipeline(model_engine, train_loader, epochs):
    model_engine.train()

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        samples_processed = 0

        # Create fresh iterator for each epoch
        train_iter = iter(train_loader)

        # For DeepSpeed pipeline, we need to use train_batch
        # The number of actual forward passes will be determined by
        # gradient_accumulation_steps and micro_batch_size
        try:
            # Let DeepSpeed handle the data iteration
            loss = model_engine.train_batch(train_iter)

            # Only the last stage has the loss
            if loss is not None and model_engine.is_last_stage():
                print(f'Epoch {epoch+1} completed, loss: {loss.item():.4f}')
            else:
                print(f'Epoch {epoch+1} completed (loss not available on this stage)')

        except StopIteration:
            print(f"StopIteration in epoch {epoch+1} - not enough data")
        except Exception as e:
            print(f"Error in epoch {epoch+1}: {e}")

        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1} completed in {epoch_time:.2f} seconds')

# Training function for non-pipelined model
def train_standard(model_engine, train_loader, epochs):
    model_engine.train()

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(model_engine.device)
            labels = labels.to(model_engine.device)

            outputs = model_engine(inputs)
            loss = F.cross_entropy(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            running_loss += loss.item()

            if i % 100 == 99:
                print(f'Epoch: {epoch+1}, Batch: {i+1}, Loss: {running_loss/100:.4f}')
                running_loss = 0.0

        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1} completed in {epoch_time:.2f} seconds')

# Evaluation function
def evaluate(model_engine, test_loader, is_pipeline=False):
    model_engine.eval()
    correct = 0
    total = 0

    if is_pipeline:
        # For pipeline parallelism, we need to handle evaluation differently
        # Because logits are only available on the last stage of the pipeline

        # Create a fresh iterator for evaluation
        test_iter = iter(test_loader)

        # Prepare data structures to collect results
        all_logits = []
        all_labels = []

        # Track whether this process is the last stage
        is_last_stage = model_engine.is_last_stage() if hasattr(model_engine, 'is_last_stage') else False

        # Collect all the labels first for reference
        all_test_labels = [batch[1] for batch in test_loader]
        test_iter = iter(test_loader)  # Reset iterator

        # Process each batch
        for i in range(len(test_loader)):
            try:
                # Process batch through pipeline
                loss, logits = model_engine.eval_batch(test_iter, return_logits=True)

                # Only the last stage will have actual logits
                if is_last_stage and logits is not None:
                    # Print for debugging
                    print(f"Batch {i}, logits shape: {logits.shape}")
                    all_logits.append(logits.detach().cpu())

            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue

        # Only the last stage computes accuracy
        if is_last_stage:
            if all_logits:
                # Combine all predictions
                all_logits = torch.cat(all_logits, dim=0)
                all_labels = torch.cat(all_test_labels, dim=0)

                print(f"All logits shape: {all_logits.shape}")
                print(f"All labels shape: {all_labels.shape}")

                # Make sure dimensions match
                min_size = min(all_logits.size(0), all_labels.size(0))
                all_logits = all_logits[:min_size]
                all_labels = all_labels[:min_size]

                # Calculate accuracy
                _, predicted = torch.max(all_logits, 1)
                total = all_labels.size(0)
                correct = (predicted == all_labels).sum().item()

                # Calculate and print accuracy
                accuracy = 100 * correct / total
                print(f'Accuracy on test set: {accuracy:.2f}%')

                # Broadcast accuracy to all stages
                if torch.distributed.is_initialized():
                    accuracy_tensor = torch.tensor([accuracy], device=model_engine.device)
                    torch.distributed.broadcast(accuracy_tensor, 0)
                    accuracy = accuracy_tensor.item()
            else:
                print("No valid logits were collected during evaluation")
                accuracy = 0
        else:
            # Non-last stages receive the accuracy
            if torch.distributed.is_initialized():
                accuracy_tensor = torch.tensor([0.0], device=model_engine.device)
                torch.distributed.broadcast(accuracy_tensor, 0)
                accuracy = accuracy_tensor.item()
                print(f'Received accuracy from last stage: {accuracy:.2f}%')
            else:
                accuracy = 0

        return accuracy
    else:
        # Standard evaluation for non-pipeline model
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(model_engine.device)
                labels = labels.to(model_engine.device)

                outputs = model_engine(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    # Calculate and print accuracy
    accuracy = 100 * correct / total if total > 0 else 0
    print(f'Accuracy on test set: {accuracy:.2f}%')
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='DeepSpeed MLP Example with Pipeline Parallelism')

    # Model parameters
    parser.add_argument('--input_dim', type=int, default=100, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=5, help='Output dimension')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    # Pipeline parameters
    parser.add_argument('--use_pipeline', action='store_true', help='Use pipeline parallelism')
    parser.add_argument('--num_stages', type=int, default=2, help='Number of pipeline stages')
    parser.add_argument('--micro_batch_size', type=int, default=8, help='Micro batch size for pipeline')

    parser.add_argument('--local_rank', type=int, default=2, help='Local rank of the process')
    parser.add_argument("--num_states", type=int, default=2, help="Number of pipeline stages")

    # DeepSpeed parameters
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    # Generate synthetic data for demonstration
    print("Generating synthetic data...")

    # Define dimensions
    input_dim = args.input_dim
    output_dim = args.output_dim
    num_train_samples = 5000
    num_test_samples = 1000

    # Generate training data
    train_inputs = torch.randn(num_train_samples, input_dim)
    # Generate random class labels
    train_labels = torch.randint(0, output_dim, (num_train_samples,))

    # Generate test data
    test_inputs = torch.randn(num_test_samples, input_dim)
    test_labels = torch.randint(0, output_dim, (num_test_samples,))

    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_inputs, test_labels)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    print(f"Created dataset with {num_train_samples} training samples and {num_test_samples} test samples")
    print(f"Input dimension: {input_dim}, Output classes: {output_dim}")

    # DeepSpeed configuration
    ds_config = {
        "train_batch_size": args.batch_size,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": args.lr
            },
        },
        "steps_per_print": 3  # Print loss every 100 steps
    }

    if args.use_pipeline:
        print("Using pipeline parallelism")
        deepspeed.init_distributed()
        # Add pipeline parallelism configuration
        ds_config["local_rank"] = args.local_rank
        ds_config["zero_optimization"] = {"stage": 0}  # Turn off ZeRO when using Pipeline
        ds_config["pipeline"] = {
            "stages": args.num_stages,
            "partition": "uniform",
            "pipe_chunk_size": args.micro_batch_size,
            "activation_checkpoint_interval": 0
        }

        # Define layer specifications for pipeline
        layers = [
            LayerSpec(MLPLayer1, args.input_dim, args.hidden_dim),
            LayerSpec(MLPLayer2, args.hidden_dim, args.output_dim)
        ]

        # Create PipelineModule with appropriate loss function
        model = PipelineModule(
            layers=layers,
            loss_fn=torch.nn.CrossEntropyLoss(),
            num_stages=args.num_stages
        )

        # Initialize DeepSpeed with pipeline
        model_engine, _, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=[p for p in model.parameters() if p.requires_grad],
            config=ds_config
        )

        # Train with pipeline parallelism
        train_pipeline(model_engine, train_loader, args.epochs)

        # Evaluate pipelined model with proper handling
        try:
            print("Starting pipeline model evaluation...")
            accuracy = evaluate(model_engine, test_loader, is_pipeline=True)
            print(f"Pipeline model evaluation complete. Accuracy: {accuracy:.2f}%")
        except Exception as e:
            import traceback
            print(f"Error during pipeline evaluation: {e}")
            print(traceback.format_exc())
    else:
        print("Using standard parallelism (no pipeline)")
        # Use ZeRO optimization for standard parallelism
        ds_config["zero_optimization"] = {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            }
        }

        # Create standard model
        model = TwoLayerMLP(args.input_dim, args.hidden_dim, args.output_dim)

        # Initialize DeepSpeed without pipeline
        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )

        # Train standard model
        train_standard(model_engine, train_loader, args.epochs)

        # Evaluate standard model
        evaluate(model_engine, test_loader)

if __name__ == "__main__":
    main()