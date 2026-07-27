import re
import sys

def parse_log(filename):
    with open(filename, 'r') as f:
        data = f.read()
    
    frames = len(re.findall(r"Tracking started\.\.\.", data))
    lost_tracking = len(re.findall(r"Fail to track local map!", data))
    resets = len(re.findall(r"Reseting active map", data))
    
    scores = []
    for match in re.finditer(r"scores0\[0\]: ([\d\.e\-]+)", data):
        scores.append(float(match.group(1)))
        
    mean_score = sum(scores) / len(scores) if scores else 0
    
    matches_count = []
    for match in re.finditer(r"matches0\[0\]: (\-?\d+)", data):
        matches_count.append(int(match.group(1)))
    
    valid_matches = len([m for m in matches_count if m != -1])
    
    print(f"Stats for {filename}:")
    print(f"Total Frames: {frames}")
    print(f"Tracking Lost: {lost_tracking} times")
    print(f"Map Resets: {resets} times")
    print(f"Mean Confidence Score (0-1): {mean_score:.4f}")
    print(f"Valid Frame-to-Frame Match Detections: {valid_matches}")

parse_log("/home/nalin/.gemini/antigravity-ide/brain/b9e5284b-9b2a-4983-958c-ed55daa6eb9b/.system_generated/tasks/task-8043.log")
