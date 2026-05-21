from hailo_platform import HEF
hef_path = "/mnt/data/base-whisper-decoder-fixed-sequence-matmul-split_h8l.hef"
hef = HEF(hef_path)
print("Inputs:")
for i in hef.get_input_vstream_infos():
    print(f"  {i.name}  shape={i.shape}  type={i.format.type}")
print("Outputs:")
for o in hef.get_output_vstream_infos():
    print(f"  {o.name}  shape={o.shape}  type={o.format.type}")
