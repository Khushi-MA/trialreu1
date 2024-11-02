from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import cv2
import mediapipe as mp
from torchvision import transforms
import numpy as np
from PIL import Image
import mediapipe as mp

# Function to perform exact match
def exact_match(query, labels):
    video_data_path = "data/video_data105.pt" 
    video_data = torch.load(video_data_path, weights_only=True)

    if query in labels:
        index = labels.index(query)
        return video_data[index]
    else:
        return None

# Function to get BERT embeddings for a word
def get_bert_embedding(word):
    inputs = BertTokenizer(word, return_tensors='pt')
    outputs = BertModel(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Function to find the most similar word using BERT
def similar_word_search(query, labels):
    query_embedding = get_bert_embedding(query)
    label_embeddings = [get_bert_embedding(label) for label in labels]

    similarities = [cosine_similarity(query_embedding, label_embedding)[0][0] for label_embedding in label_embeddings]
    max_similarity_index = np.argmax(similarities)

    if similarities[max_similarity_index] > 0.7:  # Threshold for similarity
        return labels[max_similarity_index]
    else:
        return None
    
def check_match_for_query_in_label(query, labels):
    print(f"Checking for exact match for query: '{query}'...")
    video_frames = exact_match(query, labels)

    if not video_frames:
        print(f"Exact match not found for '{query}'. Searching for similar words...")
        similar_word = similar_word_search(query, labels)
        if similar_word:
            print(f"Similar word found: '{similar_word}'. Retrieving corresponding video frames...")
            video_frames = exact_match(similar_word, labels)
            print(f"Corresponding video frames for similar word '{similar_word}' retrieved.")
        else:
            print("No similar word found.")
    else:
        print(f"Exact match found for '{query}'. Corresponding video frames retrieved.")

    return video_frames



# Function to blackout background and keep only hands and face
def blackout_background_with_hands(frame):
    print("Initializing MediaPipe Hands...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    print("Converting frame to RGB...")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(frame_rgb)

    # Create an all-black frame of the same size
    black_frame = np.zeros_like(frame)

    # Draw hand landmarks on the black frame
    if results_hands.multi_hand_landmarks:
        print("Hand landmarks detected. Drawing landmarks...")
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                black_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        print("No hand landmarks detected.")
            
    print("Returning processed frame.")
    return black_frame


# Step 1: Input Continuous Video - Extract frames from the continuous video, apply blackout, and save the video
def extract_and_blackout_frames(video_path, output_path, frame_rate=1):
    print(f"Extracting and processing frames from {video_path}...")

    cap = cv2.VideoCapture(video_path)

    # Get frame dimensions
    print("Get frame dimensions")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get the original video's FPS
    print(f"Frame width: {width}, height: {height}, FPS: {fps}")

    # Set up VideoWriter to save the processed frames into a video
    print("Set up VideoWriter to save the processed frames into a video")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    count = 0
    processed_frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            # Blackout background for the current frame (keep only hands and face)
            print("Blackout background for the current frame (keep only hands and face)")
            processed_frame = blackout_background_with_hands(frame)
            out.write(processed_frame)  # Write the processed frame to the video
            processed_frame_count += 1
            if processed_frame_count % 10 == 0:
                print(f"Processed {processed_frame_count} frames...")
        count += 1

    cap.release()
    out.release()
    print(f"Extracted and processed frames saved to {output_path}.")



# Step 2: Feature Extraction
def extract_frames(video_path, frame_rate=1):
    print(f"Extracting frames from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            frames.append(frame)
        count += 1
    cap.release()
    print(f"Extracted {len(frames)} frames from {video_path}.")
    return frames

# Normalization & Resizing
def preprocess_frame(frame, size=(224, 224)):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    frame = transform(frame)
    print("Frame of continuous videos preprocessed.")
    return frame

# Use the same feature extractor (CNN+LSTM) to extract features from the continuous video.
def extract_features_from_video(frames, model, max_frames=100):
    # frames = extract_frames(video_path)
    preprocessed_frames = [preprocess_frame(frame) for frame in frames]

    # Pad or truncate frames to max_frames
    if len(preprocessed_frames) < max_frames:
        padding = [torch.zeros_like(preprocessed_frames[0])] * (max_frames - len(preprocessed_frames))
        preprocessed_frames.extend(padding)
    else:
        preprocessed_frames = preprocessed_frames[:max_frames]

    preprocessed_frames = torch.stack(preprocessed_frames).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        features = model(preprocessed_frames)

    return features


# Step 3: Sliding Window Approach
# Slide a window over the continuous video to extract features for each segment.
def sliding_window_features(video_path, model, window_size=30, stride=15):
    frames = extract_frames(video_path)
    preprocessed_frames = [preprocess_frame(frame) for frame in frames]
    num_frames = len(preprocessed_frames)

    window_features = []
    for start in range(0, num_frames - window_size + 1, stride):
        window = preprocessed_frames[start:start + window_size]
        window = torch.stack(window).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            features = model(window)

        window_features.append((start, features))

    return window_features


# Step 4: Similarity Search
# Compute similarity scores between the segment features and the retrieved video's features.
def compute_similarity(retrieved_features, window_features):
    similarities = []
    for start, features in window_features:
        similarity = cosine_similarity(retrieved_features, features)
        similarities.append((start, similarity))
    return similarities

# Step 5: Localization
# Predict the time interval in the continuous video where the sign appears based on the similarity scores.
def find_best_match(similarities, window_size, threshold=0.8):
    best_match_start = None
    best_match_end = None
    best_score = -1

    for start, similarity in similarities:
        score = similarity[0][0]
        if score > best_score and score > threshold:
            best_score = score
            best_match_start = start
            best_match_end = start + window_size  # End frame is start + window_size

    return best_match_start, best_match_end, best_score
# def find_best_match(similarities, threshold=0.8):
#     best_match = None
#     best_score = -1
#     for start, similarity in similarities:
#         score = similarity[0][0]
#         if score > best_score and score > threshold:
#             best_score = score
#             best_match = start
#     return best_match, best_score

def check_for_presence(continuous_video_path, video_frames_from_query, black_continuous_output_path, final_continuous_output_path, model):
    if video_frames_from_query:
        retrieved_features = extract_features_from_video(video_frames_from_query, model)
        if retrieved_features is None:
            print("Failed to extract features from the retrieved video frames.")
        else:
            # Extract features from the continuous video using sliding window
            window_features = sliding_window_features(black_continuous_output_path, model, window_size=30, stride=15)

            # Compute similarity scores
            similarities = compute_similarity(retrieved_features, window_features)

            # Find the best match with start and end frames
            best_match_start, best_match_end, best_score = find_best_match(similarities, window_size=30, threshold=0.8)

            if best_match_start is not None:
                print(f"Sign found from frame {best_match_start} to frame {best_match_end} with similarity score: {best_score:.4f}")
                add_green_border_to_frames(continuous_video_path, final_continuous_output_path, best_match_start, best_match_end)
            else:
                print("Sign not found in the continuous video.")
    else:
        print("No video frames retrieved for the query.")

    # if video_frames_from_query:
    #     retrieved_features = extract_features_from_video(video_frames_from_query, model)
    #     if retrieved_features is None:
    #         print("Failed to extract features from the retrieved video frames.")
    #     else:
    #         # Extract features from the continuous video using sliding window
    #         window_features = sliding_window_features(black_continuous_output_path, model)

    #         # Compute similarity scores
    #         similarities = compute_similarity(retrieved_features, window_features)

    #         # Find the best match
    #         best_match, best_score = find_best_match(similarities)

    #         if best_match is not None:
    #             print(f"Sign found at timestamp: {best_match} frames with similarity score: {best_score:.4f}")
    #         else:
    #             print("Sign not found in the continuous video.")
    # else:
    #     print("No video frames retrieved for the query.")

def add_green_border_to_frames(video_path, output_path, start_frame, end_frame, border_thickness=10):
    print("\n\nin green border function")
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get frame properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter /object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the current frame is in the range where we want the green border
        if start_frame <= current_frame <= end_frame:
            # Add green border by drawing a rectangle around the frame
            cv2.rectangle(frame, (border_thickness, border_thickness),
                        (width-border_thickness, height-border_thickness),
                        (0, 255, 0), border_thickness)  # Green border

        # Write the frame (with or without border) to the output video
        out.write(frame)
        current_frame += 1

        # Debug: Print frame number being processed
        if current_frame % 10 == 0:  # Print every 10 frames
            print(f"Processing frame {current_frame}/{total_frames}")

    # Release everything when done
    cap.release()
    out.release()
    print(f"Video with green borders from frame {start_frame} to {end_frame} saved at {output_path}.")

