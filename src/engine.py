import os
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from timeit import default_timer as timer

def create_writer(experiment_name: str, model_name: str) -> SummaryWriter:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("runs", experiment_name, model_name, timestamp)
    return SummaryWriter(log_dir=log_dir)

def train_and_log(model, train_loader, test_loader, optimizer, loss_fn, epochs, writer, device):
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    
    # Initializing Automatic Mixed Precision (AMP) Scaler
    scaler = torch.amp.GradScaler(device_type) if device_type == "cuda" else None

    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    
    total_train_time = 0.0
    total_test_time = 0.0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch: {epoch}/{epochs}")
        
        # Training Phase
        start_train_time = timer()
        model.train()
        train_loss, train_acc = 0.0, 0.0
        
        train_prog_bar = tqdm(train_loader, desc="Training", leave=True)
        
        for X, y in train_prog_bar:
            # Asynchronous Data Transfers (non_blocking=True)
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            # Optimizing zero_grad
            optimizer.zero_grad(set_to_none=True)
            
            # Casting forward pass to float16 using AMP
            if device_type == "cuda":
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    y_pred = model(X)
                    loss = loss_fn(y_pred, y)
                
                # Scale the loss and backpropagate
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Fallback for CPU
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
            
            batch_loss = loss.item()
            batch_acc = (torch.argmax(y_pred, dim=1) == y).sum().item() / len(y)
            
            train_loss += batch_loss
            train_acc += batch_acc
            
            train_prog_bar.set_postfix(loss=batch_loss, acc=batch_acc)
            
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        end_train_time = timer()
        total_train_time += (end_train_time - start_train_time)

        # Testing Phase
        start_test_time = timer()
        model.eval()
        test_loss, test_acc = 0.0, 0.0
        
        test_prog_bar = tqdm(test_loader, desc="Testing", leave=True)
        
        with torch.inference_mode():
            for X, y in test_prog_bar:
                # Asynchronous Data Transfers (non_blocking=True)
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                
                # Applying AMP to inference for faster evaluation
                if device_type == "cuda":
                    with torch.autocast(device_type=device_type, dtype=torch.float16):
                        y_pred = model(X)
                        loss = loss_fn(y_pred, y)
                else:
                    y_pred = model(X)
                    loss = loss_fn(y_pred, y)
                
                batch_loss = loss.item()
                batch_acc = (torch.argmax(y_pred, dim=1) == y).sum().item() / len(y)
                
                test_loss += batch_loss
                test_acc += batch_acc
                
                test_prog_bar.set_postfix(loss=batch_loss, acc=batch_acc)
                
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        
        end_test_time = timer()
        total_test_time += (end_test_time - start_test_time)

        # Final epoch summary
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        writer.add_scalars("Loss", {"train": train_loss, "test": test_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "test": test_acc}, epoch)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    writer.close()
    
    return results, total_train_time, total_test_time