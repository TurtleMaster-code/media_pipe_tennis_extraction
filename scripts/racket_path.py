import cv2
import mediapipe as mp
import os
import glob
import csv

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

# List of all landmarks we want to track
LANDMARKS = {
    "right_index": mp_pose.PoseLandmark.RIGHT_INDEX,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
    "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE
}

def extract_landmarks(frame):
    """Run Mediapipe on a frame and return normalized coordinates for all landmarks."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if not results.pose_landmarks:
        return None  # No detection in this frame

    lm = results.pose_landmarks.landmark
    data = {}

    for name, idx in LANDMARKS.items():
        data[f"{name}_x"] = lm[idx].x
        data[f"{name}_y"] = lm[idx].y

    return data

# ------------------ MAIN ------------------

frames_base = r"C:\Users\arhan\OneDrive\Pictures\Documents\GitHub\Tennis_Pos_Extraction_Plus_Ai_Feedback\videos\right_side_views\frames_of_right_side_videos"
output_base = r"C:\Users\arhan\OneDrive\Pictures\Documents\GitHub\Tennis_Pos_Extraction_Plus_Ai_Feedback\videos\right_side_views\racket_path_csvs"

os.makedirs(output_base, exist_ok=True)

for i in range(1, 12):  # serve1 to serve11
    frame_dir = os.path.join(frames_base, f"right_side_serve_{i}")
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))

    csv_path = os.path.join(output_base, f"serve{i}_path.csv")

    print(f"\nProcessing serve {i}...")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Write CSV header
        header = ["frame"]
        for name in LANDMARKS.keys():
            header.append(f"{name}_x")
            header.append(f"{name}_y")
        writer.writerow(header)

        # Process each frame
        for idx, frame_path in enumerate(frames):
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            data = extract_landmarks(frame)
            if data is None:
                continue  # skip frames with no detection

            row = [idx]  # frame number
            for name in LANDMARKS.keys():
                row.append(data[f"{name}_x"])
                row.append(data[f"{name}_y"])

            writer.writerow(row)

    print(f"Saved → {csv_path}")
