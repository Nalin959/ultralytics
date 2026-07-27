import os
import re

SOURCE_DIR = 'SP_SLAM3'
DOCS_DIR = 'SP_SLAM3_Docs'

# Detailed knowledge base for core files we deeply analyzed
KNOWLEDGE_BASE = {
    'SuperPoint.cc': "This file encapsulates the SuperPoint deep learning model integration using TensorRT. It handles loading the TRT engine, allocating GPU memory for inference, pre-processing the input image, and post-processing the output tensors (confidence maps and descriptor maps) to extract keypoints and their corresponding 256-dimensional descriptors. It recently had a fix applied to remove incompatible TRT logging API usage.",
    
    'SPextractor.cc': "This file is responsible for feature extraction using the SuperPoint model in a multi-scale image pyramid. It implements the SPextractor class which manages the extraction process. A critical part of this file is `ComputeKeyPointsOctTree`, which distributes keypoints uniformly across the image using an octree structure. We recently fixed a major bug here where descriptors were being pushed to the array twice, causing a size mismatch and subsequent memory corruption during tracking.",
    
    'SPmatcher.cc': "This file implements various feature matching strategies specifically adapted for SuperPoint descriptors (which are 256D floating point vectors, unlike ORB's 32-byte binary vectors). It uses L2 distance (norm) for matching instead of Hamming distance. It includes functions like `SearchByBoW` (using Bag of Words for relocalization and loop closure), `SearchByProjection` (for tracking local map), and `SearchForTriangulation`. We added bounds checking here to catch out-of-bounds indices from corrupted BoW vectors.",
    
    'Tracking.cc': "This is the main tracking thread of the SLAM system. It handles the processing of new frames (monocular, stereo, or RGB-D), extracts features (via SPextractor), matches them against the last frame or the local map, and estimates the camera pose using g2o optimization. It manages states like OK, LOST, and handles relocalization.",
    
    'LocalMapping.cc': "This file runs in its own thread and manages the local map. It processes new KeyFrames inserted by the Tracking thread, triangulates new MapPoints, culls redundant MapPoints and KeyFrames, and performs Local Bundle Adjustment (BA) to optimize the local window of KeyFrames and MapPoints.",
    
    'LoopClosing.cc': "This file runs in a background thread and is responsible for detecting large loops in the trajectory using Bag of Words (BoW) matching (which utilizes the LightGlue or SuperPoint features mapped to a visual vocabulary). When a loop is detected, it computes a Sim3 transformation, corrects the accumulated drift, and triggers a global Pose Graph Optimization.",
    
    'MapPoint.cc': "Represents a 3D point in the world. It stores its 3D coordinates, viewing direction, and a representative descriptor (which is a SuperPoint 256D float vector). It handles thread-safe access using mutexes, notably `mMutexPos` and `mMutexFeatures`. The `isBad()` function was the site of the segmentation fault caused by the out-of-bounds array read in Tracking.",
    
    'KeyFrame.cc': "Represents a keyframe in the map, which is a snapshot of the camera pose and the features observed at that time. It stores the Bag of Words vector, the extracted features, the descriptors, and pointers to the MapPoints observed. It shares many properties with Frame but is persistent in the Map.",
    
    'Map.cc': "Manages the global map of the SLAM system, holding all KeyFrames and MapPoints. It provides thread-safe methods to add, erase, and query the map entities.",
    
    'LightGlue.cc': "Contains the integration of the LightGlue deep learning feature matcher. It loads a TensorRT engine for LightGlue, takes two sets of keypoints and descriptors, and outputs matches. It uses a thread-safe static mutex (`getInferenceMutex()`) to ensure TRT inference does not overlap and crash.",
    
    'mono_video.cc': "An example script to run the SLAM system on a monocular video file using OpenCV's VideoCapture. It initializes the SLAM system, reads frames sequentially, tracks them, and writes the output visualization (with green tracked points) to an output mp4 file."
}

def parse_cpp_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    classes = re.findall(r'class\s+(\w+)', content)
    structs = re.findall(r'struct\s+(\w+)', content)
    functions = re.findall(r'(\w+[\w\s\*&<>]+)\s+(\w+)\s*\([^)]*\)\s*\{', content)
    
    summary = []
    if classes:
        summary.append(f"Classes defined: {', '.join(set(classes))}")
    if structs:
        summary.append(f"Structs defined: {', '.join(set(structs))}")
    if functions:
        func_names = [f[1] for f in functions if f[1] not in ['if', 'while', 'for', 'switch', 'catch', 'else']]
        if func_names:
            summary.append(f"Key functions/methods implemented: {', '.join(set(func_names[:20]))}" + ("..." if len(func_names) > 20 else ""))
            
    return "\n".join(summary)

def generate_docs():
    for root, dirs, files in os.walk(SOURCE_DIR):
        # Optionally exclude thirdparty to keep it focused on the main codebase
        if 'Thirdparty' in root:
            continue
            
        rel_path = os.path.relpath(root, SOURCE_DIR)
        target_dir = os.path.join(DOCS_DIR, rel_path)
        
        os.makedirs(target_dir, exist_ok=True)
        
        for file in files:
            if not file.endswith(('.cc', '.cpp', '.h', '.hpp')):
                continue
                
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_dir, file + '.txt')
            
            with open(target_file, 'w') as f:
                f.write(f"Detailed Documentation for: {file}\n")
                f.write("=" * 40 + "\n\n")
                
                if file in KNOWLEDGE_BASE:
                    f.write("=== Deep Architecture Overview ===\n")
                    f.write(KNOWLEDGE_BASE[file] + "\n\n")
                else:
                    f.write("=== General Overview ===\n")
                    f.write(f"This file '{file}' is part of the SP_SLAM3 system.\n")
                    if file.endswith('.h') or file.endswith('.hpp'):
                        f.write("It is a header file that declares classes, structs, and function prototypes used across the system.\n\n")
                    else:
                        f.write("It is a source file that implements the core logic and algorithms for the classes defined in its corresponding header.\n\n")
                
                f.write("=== Code Structure Analysis ===\n")
                structure_info = parse_cpp_file(source_file)
                if structure_info:
                    f.write(structure_info + "\n")
                else:
                    f.write("No major classes or functions detected via simple parsing (might be a pure interface or data struct file).\n")
                    
    print(f"Documentation generated successfully in {DOCS_DIR}")

if __name__ == '__main__':
    generate_docs()
