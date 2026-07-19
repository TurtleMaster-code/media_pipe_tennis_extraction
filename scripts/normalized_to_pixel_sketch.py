import cv2
import csv
import os

# Skeleton connections (pairs of landmarks to draw lines between)
SKELETON = [
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_index"),

    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),

    ("right_shoulder", "left_shoulder"),

    ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_hip"),

    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),

    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle")
]

def load_normalized_csv(csv_path):
    """Load normalized coordinates from CSV."""
    frames_data = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames_data.append(row)

    return frames_data

def normalized_to_pixel(norm_x, norm_y, width, height):
    """Convert normalized coordinates to pixel coordinates."""
    px = int(norm_x * width)
    py = int(norm_y * height)
    return px, py

def draw_skeleton(frame, data, width, height):
    """Draw stick figure on the frame using pixel coordinates."""
    points = {}

    # Convert all normalized points to pixel coordinates
    for key in data.keys():
        if key.endswith("_x"):
            name = key[:-2]  # remove "_x"
            x = float(data[key])
            y = float(data[f"{name}_y"])
            px, py = normalized_to_pixel(x, y, width, height)
            points[name] = (px, py)

    # Draw joints
    for name, (px, py) in points.items():
        cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

    # Draw bones
    for a, b in SKELETON:
        if a in points and b in points:
            cv2.line(frame, points[a], points[b], (0, 255, 0), 2)

    return frame

# ------------------ MAIN ------------------

frames_base = r"C:\Users\arhan\OneDrive\Pictures\Documents\GitHub\Tennis_Pos_Extraction_Plus_Ai_Feedback\videos\right_side_views\frames_of_right_side_videos"
csv_base = r"C:\Users\arhan\OneDrive\Pictures\Documents\GitHub\Tennis_Pos_Extraction_Plus_Ai_Feedback\videos\right_side_views\racket_path_csvs"
output_base = r"C:\Users\arhan\OneDrive\Pictures\Documents\GitHub\Tennis_Pos_Extraction_Plus_Ai_Feedback\videos\right_side_views\stickfigure_visualizations"

os.makedirs(output_base, exist_ok=True)

for i in range(1, 12):
    print(f"\nProcessing serve {i}...")

    csv_path = os.path.join(csv_base, f"serve{i}_path.csv")
    frame_dir = os.path.join(frames_base, f"right_side_serve_{i}")
    out_dir = os.path.join(output_base, f"serve{i}_stickfigures")

    os.makedirs(out_dir, exist_ok=True)

    frames_data = load_normalized_csv(csv_path)
    frame_files = sorted(os.listdir(frame_dir))

    for idx, data in enumerate(frames_data):
        frame_path = os.path.join(frame_dir, frame_files[idx])
        frame = cv2.imread(frame_path)

        height, width = frame.shape[:2]

        frame_with_skeleton = draw_skeleton(frame, data, width, height)

        out_path = os.path.join(out_dir, f"frame_{idx}_stick.png")
        cv2.imwrite(out_path, frame_with_skeleton)

    print(f"Saved stick figures → {out_dir}")
