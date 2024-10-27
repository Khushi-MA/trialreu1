import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import create_dataloader
from model import SignRecognitionModel
from train import train_model

if __name__ == "__main__":
    # Create DataLoader and get the number of classes
    dataloader, num_classes = create_dataloader()

    # Initialize model, loss function, and optimizer
    model = SignRecognitionModel(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create the directory to save the model if it doesn't exist
    save_dir = 'data'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'sign_recognition_model1.pth')

    # Train the model and save it
    train_model(model, dataloader, criterion, optimizer, num_epochs=2, save_path=save_path)