with open('/home/nalin/gps_denied_project/SP_SLAM3/src/SPmatcher.cc', 'r') as f:
    content = f.read()

start_idx = content.find("int SPmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,")
end_idx = content.find("// --- Fallback: BoW-constrained L2 matching ---", start_idx)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + "int SPmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,\n                                      vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo)\n{\n    int nmatches=0;\n    vector<int> vMatches12(pKF1->N,-1);\n\n    " + content[end_idx:]
    with open('/home/nalin/gps_denied_project/SP_SLAM3/src/SPmatcher.cc', 'w') as f:
        f.write(content)
    print("Fixed SPmatcher successfully!")
else:
    print("Could not find blocks")
