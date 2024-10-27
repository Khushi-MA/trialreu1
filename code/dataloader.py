import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SignDataset(Dataset):
    def __init__(self, video_data, labels, transform=None, max_frames=100):
        self.video_data = video_data
        self.labels = labels
        self.transform = transform
        self.max_frames = max_frames

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):
        video = self.video_data[idx]
        label = self.labels[idx]

        # Pad or truncate video to max_frames
        if len(video) < self.max_frames:
            padding = [torch.zeros_like(video[0])] * (self.max_frames - len(video))
            video.extend(padding)
        else:
            video = video[:self.max_frames]

        if self.transform:
            video = [self.transform(frame) for frame in video]
        video = torch.stack(video)
        label = torch.tensor(label, dtype=torch.long)  # Ensure label is a tensor of type long
        return video, label

def create_dataloader(batch_size=8, max_frames=100):
    # Load preprocessed data and labels
    video_data = torch.load('data/video_databig.pt')
    labels = torch.load('data/labelsbig.pt')

    # Create a label mapping
    unique_labels = list(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    numerical_labels = [label_to_index[label] for label in labels]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = SignDataset(video_data, numerical_labels, transform=transform, max_frames=max_frames)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, len(unique_labels)