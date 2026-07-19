import cv2
import os

# Base paths
video_base = r"C:\Users\arhan\OneDrive\Pictures\Documents\GitHub\Tennis_Pos_Extraction_Plus_Ai_Feedback\videos\right_side_views\individual_right_side_videos"
output_base = r"C:\Users\arhan\OneDrive\Pictures\Documents\GitHub\Tennis_Pos_Extraction_Plus_Ai_Feedback\videos\right_side_views\frames_of_right_side_videos"

def rotate_upright(frame):
    """Force 90° clockwise rotation for iPhone videos."""
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Fix rotation
        frame = rotate_upright(frame)

        # Save frame
        out_path = os.path.join(output_folder, f"frame_{frame_idx}.png")
        cv2.imwrite(out_path, frame)

        frame_idx += 1

    cap.release()
    print(f"Extracted {frame_idx} frames → {output_folder}")

# ------------------ MAIN ------------------

for i in range(1, 12):  # 1 to 11
    video_path = os.path.join(video_base, f"serve{i}.mp4")
    output_folder = os.path.join(output_base, f"right_side_serve_{i}")

    os.makedirs(output_folder, exist_ok=True)

    print(f"\nProcessing serve{i}.mp4...")
    extract_frames(video_path, output_folder)
