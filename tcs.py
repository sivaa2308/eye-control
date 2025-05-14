import os

# Define the folder path (modify as needed)
folder_path = r"C:\github\eye-control\latest_model\test\center"

# Define the new naming pattern
new_prefix = "testing_center"

# Function to rename images sequentially
def rename_images(folder_path, new_prefix):
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist.")
        return

    # List and sort image files
    images = sorted(os.listdir(folder_path))  # Sort to maintain order
    count = 1

    for image_file in images:
        # Get file extension
        ext = os.path.splitext(image_file)[-1].lower()
        if ext not in [".jpg", ".png", ".jpeg"]:  # Process only image files
            continue

        # New file name
        new_filename = f"{new_prefix}_{count}{ext}"
        old_filepath = os.path.join(folder_path, image_file)
        new_filepath = os.path.join(folder_path, new_filename)

        # Rename file
        os.rename(old_filepath, new_filepath)
        print(f"Renamed: {old_filepath} -> {new_filepath}")

        count += 1  # Increment counter

# Run the renaming function
rename_images(folder_path, new_prefix)
