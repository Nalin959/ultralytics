with open('/home/nalin/gps_denied_project/SP_SLAM3/src/Tracking.cc', 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if "void Tracking::TrackWithMotionModel()" in line:
        print(f"TrackWithMotionModel: {i}")
    if "int ORBmatcher::SearchByProjection" in line:
        print(f"SearchByProjection: {i}")
