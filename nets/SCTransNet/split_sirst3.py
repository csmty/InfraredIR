import os
import shutil
import argparse

def organize_images(folder_A):
    # Create target subdirectories
    nuaa_dir = os.path.join(folder_A, "NUAA-SIRST")
    nudt_dir = os.path.join(folder_A, "NUDT-SIRST")
    irstd_dir = os.path.join(folder_A, "IRSTD-1K")

    os.makedirs(nuaa_dir, exist_ok=True)
    os.makedirs(nudt_dir, exist_ok=True)
    os.makedirs(irstd_dir, exist_ok=True)

    # Iterate over files in folder_A
    for filename in os.listdir(folder_A):
        file_path = os.path.join(folder_A, filename)

        # Skip directories
        if not os.path.isfile(file_path):
            continue

        # Decide target directory based on filename prefix
        if filename.startswith("XDU"):
            target_path = os.path.join(irstd_dir, filename)
        elif filename.startswith("Misc"):
            target_path = os.path.join(nuaa_dir, filename)
        else:
            target_path = os.path.join(nudt_dir, filename)

        # Move file
        shutil.move(file_path, target_path)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize images by filename prefix.")
    parser.add_argument("SIRST3_path", type=str, help="Path to SIRST3 folder")
    args = parser.parse_args()

    organize_images(args.SIRST3_path)