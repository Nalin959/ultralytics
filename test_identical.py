from compare_sam3_vs_loftr import run_sam3, run_loftr, TEST_PATH

print("----- SAM3 (Identical Image) -----")
_, _, ref_conf = run_sam3(TEST_PATH, "Test Frame (Ref)")
_, _, test_conf = run_sam3(TEST_PATH, "Test Frame (Test)")
sam3_overall = (ref_conf + test_conf) / 2 if (ref_conf + test_conf) > 0 else 0.0
print(f"SAM3 Overall Mean Confidence: {sam3_overall:.4f}")

print("\n----- EfficientLoFTR (Identical Image) -----")
_, n_matches, loftr_conf, loftr_confs = run_loftr(TEST_PATH, TEST_PATH)
print(f"EfficientLoFTR Mean Confidence: {loftr_conf:.4f}")
if len(loftr_confs) > 0:
    print(f"Max Conf: {loftr_confs.max():.4f}")
    print(f"Min Conf: {loftr_confs.min():.4f}")
