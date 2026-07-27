import os
import shutil

src_dir = "test_sequence/rgb"
src_txt = "test_sequence/rgb.txt"

dst_base = "test_sequence_euroc/mav0/cam0"
dst_data = os.path.join(dst_base, "data")
dst_csv = os.path.join(dst_base, "data.csv")

os.makedirs(dst_data, exist_ok=True)

with open(src_txt, 'r') as f_in, open(dst_csv, 'w') as f_out:
    f_out.write("#timestamp[ns],filename\n")
    for line in f_in:
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) == 2:
            ts_sec = float(parts[0])
            filename = os.path.basename(parts[1])
            ts_ns = int(ts_sec * 1e9)
            
            src_img = os.path.join(src_dir, filename)
            dst_img = os.path.join(dst_data, filename)
            
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
                f_out.write(f"{ts_ns},{filename}\n")

print(f"Converted dataset to Euroc format at {dst_base}")
