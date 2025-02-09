import os
import glob
import cv2
import torch
import argparse
import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from sam2.build_sam import build_sam2_video_predictor
import supervision as sv
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

# ========== Utility Functions ==========
def select_device():
    """Selects the computation device: CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 might give degraded performance."
        )
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def get_mask(video_path, masked_output_dir, hand_landmarker_model, sam2_predictor, margin, hand):
    os.makedirs(masked_output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error: Cannot open video {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    frame_timestamp_ms = 0  # Initialize timestamp

    # Load MediaPipe hand detector in video mode
    landmarker = load_hand_landmarker(hand_landmarker_model)

    frame_idx = 0
    while True:
        no_mask = False
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no more frames
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb_frame.shape
        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # Process detected hands
        boxes = get_bounding_box(rgb_frame, hand_landmarker_result, margin=margin, hand = hand)
        
        # Apply SAM2 segmentation on current frame
        sam2_predictor.set_image(rgb_frame)
        if boxes.shape[0] == 0:
            print(f"Target hand not detected in frame {frame_idx}")
            # print(hand_landmarker_result.hand_landmarks[idx])
            no_mask = True

        if no_mask:
            # if no chosen hand in frame, return original frame
            segmented_image = frame

        else:
            masks, scores, logits = sam2_predictor.predict(
                box=boxes,
                multimask_output=False
            )

            # Handle single vs multiple masks
            if boxes.shape[0] != 1:
                masks = np.squeeze(masks)

            # Apply mask
            mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=masks),
                mask=masks.astype(bool)
                )
            segmented_image = mask_annotator.annotate(scene=frame.copy(), detections=detections)

            
        # Save masked frame
        masked_frame_path = os.path.join(masked_output_dir, f"{frame_idx:05d}.jpg")
        cv2.imwrite(masked_frame_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
        print(f"Processed frame {frame_idx}, saved to {masked_frame_path}")

        # Update timestamp for next frame
        frame_timestamp_ms += int(1000 / frame_rate)
        frame_idx += 1

    cap.release()
    print(f"Processing complete! Frames saved in {masked_output_dir}")
    return frame_rate

def get_bounding_box(rgb_image, detection_result, margin, hand):
    MARGIN = margin
    assert hand == 'Left' or hand == 'Right'; "hand must be either 'Left' or 'Right'"
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    height, width, _ = rgb_image.shape    
    boxes = []
    for idx in range(len(detection_result.hand_landmarks)):
        hand_landmarks = detection_result.hand_landmarks[idx]
        handedness = detection_result.handedness[idx][0].category_name  # "Left" or "Right"

        if handedness != hand:  # Only segment left hands
            continue

        # Convert landmarks to pixel coordinates
        x_coordinates = np.array([lm.x * width for lm in hand_landmarks], dtype=np.int32)
        y_coordinates = np.array([lm.y * height for lm in hand_landmarks], dtype=np.int32)

        # Compute bounding box with margin
        x_min, y_min = np.min(x_coordinates) - MARGIN, np.min(y_coordinates) - MARGIN
        x_max, y_max = np.max(x_coordinates) + MARGIN, np.max(y_coordinates) + MARGIN

        # Ensure bounding box stays within image bounds
        x_min, y_min = max(x_min, 0), max(y_min, 0)
        x_max, y_max = min(x_max, width), min(y_max, height)

        boxes.append([x_min, y_min, x_max, y_max])

        # Convert to NumPy array
    boxes = np.array(boxes)
    return boxes


def load_hand_landmarker(model_path):
    """Creates a HandLandmarker object."""
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO, num_hands=2,
        min_hand_presence_confidence = 0.2)
    landmarker = HandLandmarker.create_from_options(options)
    return landmarker



def create_video_from_frames(image_folder, frame_rate=30):
    """Creates a video from a sequence of images."""
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.endswith(".jpg")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    if not image_files:
        raise ValueError("No images found in the folder.")

    first_frame = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video_path = os.path.join(image_folder, 'masked_video.mp4')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for image_file in image_files:
        frame = cv2.imread(os.path.join(image_folder, image_file))
        video_writer.write(frame)

    video_writer.release()
    return

# ========== Main Pipeline ==========

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process video with optional margins and hand selection.")
    parser.add_argument('--margin', type=int, default=30, help='Margin (in pixels) around the hand detection for bounding box (default: 30)')
    parser.add_argument('--hand', type=str, default='Left', choices=['Left', 'Right'], help='Hand to target (default: Left)')
    
    args = parser.parse_args()

    # Paths and directories
    video_files = glob.glob('../source_video/*.mp4')
    if len(video_files) == 0 or len(video_files) > 1:
        raise ValueError("Please provide exactly one video file in the source_video directory.")
        
    video = video_files[0]
    src_dir = "../source_video/"
    masked_output_dir = "../masked_video/"
    hand_landmarker_model = "hand_landmarker.task"
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # Select device
    device = select_device()

    # load SAM2 image module
    print("Loading SAM2 model...")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    predictor = SAM2ImagePredictor(sam2_model)  
    print("SAM2 model loaded!")

    print(f"Processing video {video}...")
    frame_rate = get_mask(video, masked_output_dir, hand_landmarker_model, predictor, margin = args.margin, hand = args.hand)
    create_video_from_frames(masked_output_dir, frame_rate=frame_rate)
    print(f"Video saved to {masked_output_dir}")
    
if __name__ == "__main__":
    main()