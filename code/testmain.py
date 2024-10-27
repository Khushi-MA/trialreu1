import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import create_dataloader
from model import SignRecognitionModel
from test import check_match_for_query_in_label, extract_and_blackout_frames, check_for_presence

if __name__ == "__main__":
    print("Creating DataLoader and getting the number of classes...")
    dataloader, num_classes = create_dataloader()
    print("DataLoader created and number of classes obtained.")

    print("Loading the model...")
    model_path = r"data/sign_recognition_model_10epoch.pth"
    model = SignRecognitionModel(num_classes)  # Instantiate the model with the correct class

    # Load the model weights with device compatibility
    device = torch.device("cpu")  # Change to "cuda" if GPU is available
    # model.load_state_dict(torch.load(model_path, map_location=device))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))


    # Set the model to evaluation mode for inference
    model.eval()
    print("Model loaded and set to evaluation mode.")


    print("Loading preprocessed data and labels...")
    video_data_path = "data/video_databig.pt"  # Replace with the actual path to labels.pt
    video_data = torch.load(video_data_path, weights_only=True)
    labels_path = "data/labelsbig.pt"  # Replace with the actual path to labels.pt
    labels = torch.load(labels_path, weights_only=True)
    print("Data and labels loaded.")

    query = "butcher"
    print(f"Processing query: {query}")
    video_frames_from_query = check_match_for_query_in_label(query, labels)
    print("Query processed and matching frames obtained.")

    # Example usage:
    # Step 1: Input Continuous Video - Extract frames from the continuous video, apply blackout, and save the video
    print("Assigning paths.")
    continuous_video_path = 'results/v0.mp4'
    black_continuous_output_path = 'results/v0-black.mp4'
    final_continuous_output_path = 'results/v0-greeborder.mp4'

    print("Extracting frames from the continuous video and applying blackout...")
    extract_and_blackout_frames(continuous_video_path, black_continuous_output_path, frame_rate=1)
    print("Continuous video frames extracted and saved.")
    
    # continuous_video_path = black_continuous_output_path
    print("Checking for presence of query frames in the continuous video...")
    check_for_presence(continuous_video_path, video_frames_from_query, black_continuous_output_path, final_continuous_output_path, model)
    print("Presence check completed and final video saved.")