import cv2
import mediapipe as mp
import numpy as np
import glob
import csv
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

def detect_highest_wrist(video_path):
    cap = cv2.VideoCapture(video_path)

    wrist_positions = []
    frame_indices = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Downscale for speed
        frame_resized = cv2.resize(frame, (640, 360))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

            wrist_positions.append(wrist.y)
            frame_indices.append(frame_idx)

        frame_idx += 1

    cap.release()

    if not wrist_positions:
        return None

    # Highest wrist = minimum y value (MediaPipe y: 0 top, 1 bottom)
    highest_idx = frame_indices[np.argmin(wrist_positions)]
    return int(highest_idx)

def save_frame(video_path, frame_number, output_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        # Force 90° clockwise rotation to fix iPhone sideways issue
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(output_path, frame)
    cap.release()

# ------------------ MAIN ------------------

video_folder = "videos/right_side_views/individual_right_side_videos/*.mp4"
output_folder = "videos/right_side_views/individual_right_side_contact_frames"
os.makedirs(output_folder, exist_ok=True)

output_csv = os.path.join(output_folder, "highest_wrist_frames.csv")

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["video_name", "highest_wrist_frame", "saved_image"])

    for vid in glob.glob(video_folder):
        highest_frame = detect_highest_wrist(vid)

        output_image = os.path.join(
            output_folder,
            os.path.basename(vid) + "_highest_wrist.png"
        )

        save_frame(vid, highest_frame, output_image)

        writer.writerow([os.path.basename(vid), highest_frame, output_image])
        print(f"{os.path.basename(vid)} → highest wrist frame {highest_frame} → saved {output_image}")
