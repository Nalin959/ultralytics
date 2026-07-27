import re
with open('/home/nalin/gps_denied_project/SP_SLAM3/src/Tracking.cc', 'r') as f:
    content = f.read()

# Fix pLG->match calls to pass imgSize twice
content = content.replace("imgSize);", "imgSize, imgSize);")

with open('/home/nalin/gps_denied_project/SP_SLAM3/src/Tracking.cc', 'w') as f:
    f.write(content)
print("Patched successfully!")
