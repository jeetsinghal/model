import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

STATUS_FILE = "training_status.json"
HISTORY_FILE = "training_history.json"
CHECKPOINT_FILE = "best_crop_model.pth"
CLASSES_FILE = "class_names.json"

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def write_status(data):
    try:
        temp_file = f"{STATUS_FILE}.tmp"
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_file, STATUS_FILE)
    except Exception as e:
        print(f"Warning: Failed to write status: {e}", file=sys.stderr)

def train_model(epochs=10, batch_size=32, lr=3e-4, resume=True):
    device = get_device()
    print(f"==================================================")
    print(f"  AgriVision AI — Deep Neural Training Engine")
    print(f"  Target Epochs : {epochs}")
    print(f"  Batch Size    : {batch_size}")
    print(f"  Learning Rate : {lr}")
    print(f"  Hardware Accel: {device}")
    print(f"==================================================")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dir = "Train"
    val_dir = "Validation"
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Missing '{train_dir}' or '{val_dir}' datasets in current directory.")
        
    print("Loading datasets from disk...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Dataset verified: {num_classes} classes, {len(train_dataset)} train samples, {len(val_dataset)} validation samples.")
    
    with open(CLASSES_FILE, "w") as f:
        json.dump(class_names, f, indent=2)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    
    print("Initializing MobileNetV3-Large neural backbone...")
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)
    
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)
    
    start_epoch = 1
    best_val_acc = 0.0
    history = []
    
    # Resume from existing checkpoint if available
    if resume and os.path.exists(CHECKPOINT_FILE):
        try:
            print(f"Loading weights from {CHECKPOINT_FILE} for continuation...")
            checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                prev_epoch = checkpoint.get("epoch", 0)
                best_val_acc = checkpoint.get("val_acc", 0.0)
                start_epoch = prev_epoch + 1
                print(f"==> Successfully resumed from Epoch {prev_epoch} (Saved Val Acc: {best_val_acc*100:.2f}%)")
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Warning: Could not resume checkpoint: {e}")
            
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r") as f:
                    th = json.load(f)
                    history = th.get("history", [])
                    if "best_val_acc" in th:
                        val_pct = th["best_val_acc"]
                        best_val_acc = val_pct / 100.0 if val_pct > 1.0 else val_pct
            except Exception as e:
                print(f"Warning: Could not read history file: {e}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Fast forward scheduler if resuming
    for _ in range(1, start_epoch):
        scheduler.step()
        
    total_batches = len(train_loader)
    
    if start_epoch > epochs:
        print(f"Model is already trained for {start_epoch - 1} epochs (>= target {epochs}).")
        write_status({
            "is_training": False,
            "completed": True,
            "current_epoch": start_epoch - 1,
            "total_epochs": epochs,
            "best_val_acc": round(best_val_acc * 100, 2),
            "message": f"Already reached {epochs} epochs! Best Val Accuracy: {round(best_val_acc * 100, 2)}%"
        })
        return
        
    print(f"\n🚀 Commencing Training: Epoch {start_epoch} to {epochs}")
    write_status({
        "is_training": True,
        "completed": False,
        "current_epoch": start_epoch,
        "total_epochs": epochs,
        "batch": 0,
        "total_batches": total_batches,
        "epoch_progress_pct": 0.0,
        "best_val_acc": round(best_val_acc * 100, 2),
        "status_text": f"Starting Epoch {start_epoch} of {epochs}...",
        "updated_at": time.time()
    })
    
    start_total_time = time.time()
    
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)
            
            # Periodic status update for UI live feedback
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == total_batches:
                batch_acc = running_corrects / total_samples
                batch_loss = running_loss / total_samples
                pct = round(((batch_idx + 1) / total_batches) * 100, 1)
                
                write_status({
                    "is_training": True,
                    "completed": False,
                    "current_epoch": epoch,
                    "total_epochs": epochs,
                    "batch": batch_idx + 1,
                    "total_batches": total_batches,
                    "epoch_progress_pct": pct,
                    "running_train_acc": round(batch_acc * 100, 2),
                    "running_train_loss": round(batch_loss, 4),
                    "best_val_acc": round(best_val_acc * 100, 2),
                    "status_text": f"Epoch {epoch}/{epochs} | Batch {batch_idx+1}/{total_batches} ({pct}%) | Loss: {batch_loss:.4f} | Acc: {batch_acc*100:.1f}%",
                    "updated_at": time.time()
                })
                
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx+1}/{total_batches}] "
                      f"Loss: {batch_loss:.4f} | Acc: {batch_acc*100:.2f}%")
        
        scheduler.step()
        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = running_corrects / total_samples
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0
        
        write_status({
            "is_training": True,
            "completed": False,
            "current_epoch": epoch,
            "total_epochs": epochs,
            "status_text": f"Validating Epoch {epoch}/{epochs} against {len(val_dataset)} test samples...",
            "best_val_acc": round(best_val_acc * 100, 2),
            "updated_at": time.time()
        })
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()
                val_samples += inputs.size(0)
                
        epoch_val_loss = val_loss / val_samples
        epoch_val_acc = val_corrects / val_samples
        epoch_time = time.time() - epoch_start
        
        print(f"\n=> Epoch {epoch}/{epochs} Finished in {epoch_time:.1f}s")
        print(f"   Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}%")
        print(f"   Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc*100:.2f}%\n")
        
        # Append or replace epoch entry in history
        history = [h for h in history if h.get("epoch") != epoch]
        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc * 100, 2),
            "val_loss": round(epoch_val_loss, 4),
            "val_acc": round(epoch_val_acc * 100, 2),
            "duration_sec": round(epoch_time, 1)
        })
        history.sort(key=lambda x: x["epoch"])
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": epoch_val_acc,
                "val_loss": epoch_val_loss,
                "architecture": "mobilenet_v3_large"
            }, CHECKPOINT_FILE)
            print(f"   🌟 New best model saved! Validation Acc: {epoch_val_acc*100:.2f}%\n")
            
        with open(HISTORY_FILE, "w") as f:
            json.dump({
                "best_val_acc": round(best_val_acc * 100, 2),
                "total_epochs": epoch,
                "history": history
            }, f, indent=2)
            
    total_duration = time.time() - start_total_time
    print(f"\n🎉 10-Epoch Training Complete! Total time: {total_duration/60:.2f} mins. Peak Val Accuracy: {best_val_acc*100:.2f}%")
    
    write_status({
        "is_training": False,
        "completed": True,
        "current_epoch": epochs,
        "total_epochs": epochs,
        "best_val_acc": round(best_val_acc * 100, 2),
        "status_text": f"Training completed successfully! Peak Validation Accuracy: {round(best_val_acc * 100, 2)}%",
        "history": history,
        "updated_at": time.time()
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriVision AI 10-Epoch Model Training Engine")
    parser.add_argument("--epochs", type=int, default=10, help="Total target epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh without loading existing checkpoint")
    args = parser.parse_args()
    
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=not args.no_resume
    )
