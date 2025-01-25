import os
import cv2
import torch
import subprocess
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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


def extract_frames(video_path, output_dir):
    """Extracts frames from a video and saves them as JPEG images."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error: Cannot open video {video_path}")

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"{frame_index:05d}.jpg")
        cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_index += 1

    cap.release()
    print(f"Frames extracted to {output_dir}")


def draw_landmarks_on_image(rgb_image, detection_result):
  MARGIN = 10  # pixels
  FONT_SIZE = 1
  FONT_THICKNESS = 1
  HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


def load_hand_landmarker(model_path):
    """Creates a HandLandmarker object."""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    return vision.HandLandmarker.create_from_options(options)


def show_mask(mask, ax, obj_id=None):
    """Visualizes a mask on the current axes."""
    cmap = plt.get_cmap("tab10")
    color = np.array([*cmap(obj_id % 10)[:3], 0.6])  # Unique color for each obj_id
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def generate_frames_from_video(video_segments, frames, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through the frames and save them with masks
    for out_frame_idx in range(0, len(frames)):
        # Create a figure without axis
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis('off')  # Turn off axes

        # Load the frame
        frame_path = os.path.join(input_dir, frames[out_frame_idx])
        frame_image = Image.open(frame_path)
        ax.imshow(frame_image)

        # Apply the masks
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, ax, obj_id=out_obj_id)

        # Save the frame with mask as a JPEG
        masked_frame_path = os.path.join(output_dir, f"{out_frame_idx:05d}.jpg")
        fig.savefig(masked_frame_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)  # Close the figure to free memory

    print(f"All masked frames have been saved to: {output_dir}")

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
    print(f"Video saved to {output_video_path}")

# ========== Main Pipeline ==========

def part1(output_dir, detector):
    frame_names = [
        p for p in os.listdir(output_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    frame_idx = 0
    frame_path = os.path.join(output_dir, frame_names[frame_idx])

    # Load the first image.
    image = mp.Image.create_from_file(frame_path)

    # Detect hand landmarks from the input image.
    detection_result = detector.detect(image)
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    # Process the output of the hand landmark model to be compatible with SAM2 prompt
    width = annotated_image.shape[1]
    height = annotated_image.shape[0]
    clicks_array_r = []
    clicks_array_l = []
    for landmark in detection_result.hand_landmarks[0]: # right hand
        # un-normalized coordinates
        clicks_array_r.append([landmark.x * width, landmark.y * height])
    clicks_array_r = np.array(clicks_array_r, dtype=np.float32)

    for landmark in detection_result.hand_landmarks[1]: # left hand
        # un-normalized coordinates
        clicks_array_l.append([landmark.x * width, landmark.y * height])
    clicks_array_l = np.array(clicks_array_l, dtype=np.float32)
    return clicks_array_r, clicks_array_l, frame_names

def part2(predictor, inference_state, frame_names, output_dir, clicks_array_r, clicks_array_l, masked_output_dir):
    prompts = {}
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1 # give a unique id to right hand

    neg_points = np.array([[420, 320], [405, 335], [425, 375], [450, 390]], dtype=np.float32)
    points = np.vstack([clicks_array_r, neg_points])

    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1 for _ in range(len(clicks_array_r))], np.int32)

    neg_labels = np.array([0, 0, 0, 0], np.int32)
    labels = np.hstack([labels, neg_labels])

    prompts[ann_obj_id] = points, labels
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    ann_obj_id = 2  # give a unique id to left hand
    neg_points = np.array([[900, 570], [880, 580], [835, 580]], dtype=np.float32)
    points = np.vstack([clicks_array_l, neg_points])

    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1 for _ in range(len(clicks_array_l))], np.int32)

    neg_labels = np.array([0, 0, 0], np.int32)
    labels = np.hstack([labels, neg_labels])

    prompts[ann_obj_id] = points, labels

    # `add_new_points_or_box` returns masks for all objects added so far on this interacted frame
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    generate_frames_from_video(video_segments, frame_names, output_dir, masked_output_dir)
    create_video_from_frames(masked_output_dir, frame_rate=30)


def main():
    # Paths and directories
    from sam2.build_sam import build_sam2_video_predictor
    video = '../source_video/test.mp4'
    output_dir = "../source_video/"
    masked_output_dir = "../masked_video/"
    hand_landmarker_model = "hand_landmarker.task"
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # Select device
    device = select_device()

    # Extract frames from the video
    extract_frames(video, output_dir)

    # Load HandLandmarker
    detector = load_hand_landmarker(hand_landmarker_model)

    # Initialize SAM2 video predictor
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(video_path=output_dir)
    # PART 1: Process frames and generate mask for first frame
    clicks_array_r, clicks_array_l, frame_names = part1(output_dir, detector)

    # PART 2: Using outout from Part 1, process frames and generate mask for all frames
    part2(predictor, inference_state, frame_names, output_dir, clicks_array_r, clicks_array_l, masked_output_dir)
    
if __name__ == "__main__":
    main()