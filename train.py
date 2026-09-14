import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import cv2
import os
import json
import config
import argparse


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class PlantDiseaseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()


    def _make_dataset(self):
        samples = []
        for target_class in self.classes:
            target_dir = os.path.join(self.data_dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, self.class_to_idx[target_class])
                    samples.append(item)
        return samples


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = cv2.imread(path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)


        # Convert to tensor and normalize
        if self.transform:
            sample = self.transform(sample)
        return sample, target


def get_model(model_name, num_classes, base_weights_path=None, freeze_until=None):


    """


    Initializes a model, optionally loading custom base weights and freezing layers.


   


    Args:


        model_name (str): The name of the model architecture.


        num_classes (int): The number of output classes for the final layer.


        base_weights_path (str, optional): Path to a .pth file with custom pre-trained weights.


        freeze_until (str, optional): Name of the last layer to freeze. All subsequent


                                      layers will be trainable.






    Returns:


        torch.nn.Module: The initialized model.


    """


    # Initialize a torchvision model with ImageNet weights.


    if model_name == 'resnet34':


        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)


    elif model_name == 'resnet50':


        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)


    elif model_name == 'mobilenet_v2':


        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)


    else:


        raise ValueError(f"Model {model_name} not supported.")






    # 2. Load custom base weights if a path is provided


    if base_weights_path:


        print(f"Loading custom base weights from: {base_weights_path}")


        model.load_state_dict(torch.load(base_weights_path), strict=False)






    # 3. Freeze layers up to the specified layer


    if freeze_until:


        print(f"Freezing layers until (and including): {freeze_until}")


        freezing = True


        for name, param in model.named_parameters():


            if name.startswith(freeze_until):


                freezing = False


           


            if freezing:


                param.requires_grad = False


            else:


                param.requires_grad = True


   


    # 4. Replace the final classification layer for the new task


    if model_name in ['resnet34', 'resnet50']:


        # The new layer will have requires_grad=True by default


        model.fc = nn.Linear(model.fc.in_features, num_classes)


    elif model_name == 'mobilenet_v2':


        # The new layer will have requires_grad=True by default


        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)


   


    return model


def train(data_dir, model_name, epochs, lr, batch_size, model_path, model_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])


    train_dataset = PlantDiseaseDataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = PlantDiseaseDataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
   
    num_classes = len(train_dataset.classes)
    # Pass the base model path and freeze setting from config to the model getter
    model = get_model(
        model_name,
        num_classes,
        base_weights_path=config.TrainingConfig.BASE_MODEL_PATH,
        freeze_until=config.TrainingConfig.FREEZE_LAYERS_UNTIL,
    )
    model.to(DEVICE)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    # Initialize the learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.TrainingConfig.LR_SCHEDULER_STEP_SIZE,
        gamma=config.TrainingConfig.LR_SCHEDULER_GAMMA
    )


    # Initialize early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0


    # Lists to store loss history for plotting
    train_loss_history = []
    val_loss_history = []


    best_val_acc = 0.0


    print(f"Starting training with {model_name} on {DEVICE}.")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
           
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
           
            running_loss += loss.item() * images.size(0)
           
        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)
       
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
               
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_loss_history.append(val_epoch_loss)
        val_acc = correct / total
       
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {epoch_loss:.3f}.. "
              f"Val loss: {val_epoch_loss:.3f}.. "
              f"Val accuracy: {val_acc:.3f}")

        # Step the scheduler
        scheduler.step()


        # Save the model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved to {model_path} with accuracy: {best_val_acc:.3f}")


        # Early stopping check
        if config.TrainingConfig.APPLY_EARLY_STOPPING:
            if val_epoch_loss < best_val_loss - config.TrainingConfig.EARLY_STOPPING_MIN_DELTA:
                best_val_loss = val_epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"Early stopping counter: {epochs_no_improve}/{config.TrainingConfig.EARLY_STOPPING_PATIENCE}")


            if epochs_no_improve >= config.TrainingConfig.EARLY_STOPPING_PATIENCE:
                print(f"Stopping early as validation loss did not improve for {config.TrainingConfig.EARLY_STOPPING_PATIENCE} epochs.")
                break


    # After the loop, plot and save the loss graph
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)


    plot_path = os.path.join(model_dir, 'loss_plot.png')
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")
    plt.close()


    # Save class_to_idx mapping
    class_to_idx_path = os.path.splitext(model_path)[0] + '_class_to_idx.json'
    with open(class_to_idx_path, 'w') as f:
        json.dump(train_dataset.class_to_idx, f)
    print(f"Class to index mapping saved to {class_to_idx_path}")




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--data_dir', type=str, required=False, default=None, help='Directory with train and val data')
    parser.add_argument('--processed_dir', type=str, default=None, help='Path to processed dataset containing train/val/test subfolders (overrides --data_dir if provided)')
    parser.add_argument('--model_name', type=str, default=config.TrainingConfig.MODEL_NAME, choices=config.TrainingConfig.AVAILABLE_MODELS, help='Model to train')
    parser.add_argument('--epochs', type=int, default=config.TrainingConfig.EPOCHS, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=config.TrainingConfig.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=config.TrainingConfig.BATCH_SIZE, help='Batch size')
    parser.add_argument('--model_path', type=str, required=False, default=None, help='Path to save the trained model (will default to ./models/trained_model.pth)')
    args = parser.parse_args()
   
    # Require an explicit data directory so no machine-specific paths are embedded.
    if not args.data_dir and not args.processed_dir:
        parser.error("Provide --processed_dir (or --data_dir) containing train/ and val/ folders")


    # If a processed_dir is provided, use that as the data root (it overrides data_dir)
    data_root = args.processed_dir if args.processed_dir else args.data_dir


    # Basic validation: ensure data_root and required subfolders exist
    if not os.path.isdir(data_root):
        parser.error(f"Data directory does not exist: {data_root}")


    # Require at least 'train' and 'val' subfolders
    train_sub = os.path.join(data_root, 'train')
    val_sub = os.path.join(data_root, 'val')
    if not os.path.isdir(train_sub):
        parser.error(f"Required subdirectory missing: {train_sub}")
    if not os.path.isdir(val_sub):
        parser.error(f"Required subdirectory missing: {val_sub}")


    # If model_path wasn't provided, default to ./models/trained_model.pth
    if not args.model_path:
        default_model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(default_model_dir, exist_ok=True)
        args.model_path = os.path.join(default_model_dir, 'trained_model.pth')
        print(f"No --model_path provided. Defaulting to: {args.model_path}")


    # Ensure we pass a model_dir to the train() function (used to save plots)
    model_dir = os.path.dirname(args.model_path) if os.path.dirname(args.model_path) else '.'
    train(data_root, args.model_name, args.epochs, args.lr, args.batch_size, args.model_path, model_dir)


