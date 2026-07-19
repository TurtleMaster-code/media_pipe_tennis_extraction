import cv2
import mediapipe as mp
import numpy as np
import os
import glob
import csv

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

def detect_highest_arm(frames):
    """Find frame where arm/wrist/elbow/shoulder is highest."""
    best_frame = None
    best_height = 999  # smallest y = highest point

    for frame_path in frames:
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if not results.pose_landmarks:
            continue

        lm = results.pose_landmarks.landmark

        wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST].y
        elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y
        shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

        # Use the minimum of all three (highest point)
        height = min(wrist, elbow, shoulder)

        if height < best_height:
            best_height = height
            best_frame = frame_path

    return best_frame

# ------------------ MAIN ------------------

frames_base = r"C:\Users\arhan\OneDrive\Pictures\Documents\GitHub\Tennis_Pos_Extraction_Plus_Ai_Feedback\videos\right_side_views\frames_of_right_side_videos"
output_base = r"C:\Users\arhan\OneDrive\Pictures\Documents\GitHub\Tennis_Pos_Extraction_Plus_Ai_Feedback\videos\right_side_views\1st_try_frame_analysis_highest_arm"

os.makedirs(output_base, exist_ok=True)

csv_path = os.path.join(output_base, "1st_try_frame_analysis_highest_arm.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["serve_number", "highest_arm_frame", "saved_image"])

    for i in range(1, 12):  # serve1 to serve11
        frame_dir = os.path.join(frames_base, f"right_side_serve_{i}")
        frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))

        print(f"\nProcessing right_side_serve_{i}...")

        best_frame = detect_highest_arm(frames)

        if best_frame is None:
            print(f"No pose detected for serve {i}.")
            writer.writerow([i, "None", "None"])
            continue

        # Load frame (already upright from Script 1)
        frame = cv2.imread(best_frame)

        # Save highest-arm frame
        out_path = os.path.join(output_base, f"serve{i}_highest_arm.png")
        cv2.imwrite(out_path, frame)

        writer.writerow([i, best_frame, out_path])
        print(f"Highest arm frame saved → {out_path}")
