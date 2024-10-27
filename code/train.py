import os
import torch
import torch.optim as optim
import torch.nn as nn
from dataloader import create_dataloader
from model import SignRecognitionModel

def train_model(model, dataloader, criterion, optimizer, num_epochs=2, save_path=None):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f"Starting epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)} processed. Current loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs} completed. Loss: {epoch_loss:.4f}")

    # Save the trained model at the specified path
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
