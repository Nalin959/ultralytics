import re
with open('/home/nalin/gps_denied_project/SP_SLAM3/src/Tracking.cc', 'r') as f:
    content = f.read()

# Find MonocularInitialization
start_idx = content.find("void Tracking::MonocularInitialization()")
if start_idx != -1:
    end_idx = content.find("void Tracking::StereoInitialization()", start_idx)
    section = content[start_idx:end_idx]
    
    # Replace nmatches<100 with nmatches<30
    patched_section = section.replace("nmatches<100", "nmatches<30")
    
    content = content[:start_idx] + patched_section + content[end_idx:]
    with open('/home/nalin/gps_denied_project/SP_SLAM3/src/Tracking.cc', 'w') as f:
        f.write(content)
    print("Patched successfully!")
else:
    print("Could not find MonocularInitialization")
