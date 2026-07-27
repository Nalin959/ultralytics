import re
with open('/home/nalin/gps_denied_project/SP_SLAM3/src/SPmatcher.cc', 'r') as f:
    content = f.read()

# Find SearchForTriangulation
start_idx = content.find("int SPmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,")
if start_idx != -1:
    end_idx = content.find("int SPmatcher::SearchBySim3(", start_idx)
    section = content[start_idx:end_idx]
    
    # Remove the LightGlue block
    lg_start = section.find("// --- LightGlue path ---")
    lg_end = section.find("// --- Fallback: BoW-constrained L2 matching ---")
    
    if lg_start != -1 and lg_end != -1:
        patched_section = section[:lg_start] + section[lg_end:]
        content = content[:start_idx] + patched_section + content[end_idx:]
        with open('/home/nalin/gps_denied_project/SP_SLAM3/src/SPmatcher.cc', 'w') as f:
            f.write(content)
        print("Patched SPmatcher successfully!")
    else:
        print("Could not find LightGlue block in SearchForTriangulation")
else:
    print("Could not find SearchForTriangulation")
