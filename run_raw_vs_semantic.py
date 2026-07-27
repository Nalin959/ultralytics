import test_efficient_loftr
import numpy as np

# Let's run LoFTR between the raw test_frame and its semantic overlay
print("Running LoFTR on Raw vs Semantic Overlay...")
test_efficient_loftr.run_test(
    "test_frame.png",
    "/home/nalin/.gemini/antigravity/brain/3b3f5db5-29ff-4a4e-a016-88d87880605f/semantic_expanded_test_frame_overlay.jpg",
    "raw_vs_semantic_match.jpg",
    "Raw vs Semantic Overlay",
    rotate_img0_deg=0
)
