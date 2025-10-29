import os

def rename_images_with_suffix(folder_path: str, suffix: str):
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}

    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        name, ext = os.path.splitext(filename)

        # Only process image files
        if ext.lower() in image_extensions:
            new_name = f"{suffix}{name}{ext}"
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)

            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_name}")

# Example usage:
rename_images_with_suffix(".", "image_2_")
