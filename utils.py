import os
def manual_tracking(
        output: dict,
        file_name: str,
        tracking_dir: str = "manual_tracking/"
        ) -> None:
    file_path = tracking_dir + file_name
    if file_name in  os.listdir(tracking_dir):
        mode = "a"
    else:
        mode = "x"
    print(mode)
    with open(file_path, mode=mode) as f:
        for metadata, value in output.items():
            f.writelines(f"{metadata}: {value}")
            f.writelines("\n")
        f.writelines("-"*48)
        f.writelines("\n")
    f.close()
