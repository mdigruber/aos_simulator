import os
import numpy as np
from PIL import Image
from tifffile import imwrite

def convert_32bit_to_16bit(image_path, output_path):
    """
    Convert a 32-bit image to 16-bit while normalizing values to fit the 16-bit range.

    Args:
        image_path (str): Path to the input 32-bit image.
        output_path (str): Path to save the 16-bit image.

    Returns:
        None
    """
    # Load the 32-bit image as a NumPy array
    image = Image.open(image_path)
    
    # Get the dimensions of the original image
    width, height = image.size

    # Calculate the center coordinates
    center_x, center_y = width // 2, height // 2

    # Define the cropping box
    crop_width, crop_height = 512, 512
    x_start = center_x - (crop_width // 2)
    y_start = center_y - (crop_height // 2)
    x_end = x_start + crop_width
    y_end = y_start + crop_height

    # Crop the image
    cropped_image = image.crop((x_start, y_start, x_end, y_end))

    image = cropped_image
    
    image_data = (np.array(image, dtype=np.float32) + 273.15) * 100  # 0 - 65535

    # Normalize the 32-bit image to the range [0, 65535]
    min_val = np.min(image_data)
    max_val = np.max(image_data)

    min_index = np.argmin(image_data)
    max_index = np.argmax(image_data)
    
    # Convert flattened indices to 2D indices
    min_index = np.unravel_index(min_index, image_data.shape)
    max_index = np.unravel_index(max_index, image_data.shape)

    print(f"Min index: {min_index}, Max index: {max_index}")
    print(f"Min value: {image_data[min_index]}, Max value: {image_data[max_index]}")

    original_size = (512, 512)
    padded_size = (824, 824)

    # Calculate padding sizes
    pad_top = (padded_size[0] - original_size[0]) // 2
    pad_bottom = padded_size[0] - original_size[0] - pad_top
    pad_left = (padded_size[1] - original_size[1]) // 2
    pad_right = padded_size[1] - original_size[1] - pad_left

    # Pad the image with zeros (black)
    padded_image = np.pad(
        image_data.astype(np.uint16),
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0
    )
    print(f"Original shape: {image_data.shape}, Padded shape: {padded_image.shape}")
    print(f"Padded image dtype: {padded_image.dtype}")

    # Convert to 16-bit integers
    image_16bit = padded_image.astype(np.uint16) 
    
    # Rotate the image 90 degrees to the right (clockwise)
    image_rotated = np.rot90(image_16bit, k=-1)  # k=-1 for clockwise rotation

    print(f"Value at min index after rotation: {image_16bit[min_index]}")
    print(f"Value at max index after rotation: {image_16bit[max_index]}")

    image_transformed = np.flipud(image_rotated)

    # Save the image
    imwrite(output_path, image_transformed)

    print(f"32-bit image converted to 16-bit and saved to {output_path}")

def convert_folder(input_root_folder, output_root_folder):
    """
    Traverse the input_root_folder and convert all .tif images in subfolders to .png.

    Args:
        input_root_folder (str): Path to the root folder containing subfolders with .tif images.
        output_root_folder (str): Path to the root folder where converted .png images will be saved.

    Returns:
        None
    """
    for root, dirs, files in os.walk(input_root_folder):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                input_image_path = os.path.join(root, file)
                
                # Construct the relative path to maintain folder structure
                relative_path = os.path.relpath(root, input_root_folder)
                output_folder = os.path.join(output_root_folder, relative_path)
                
                # Create the output subfolder if it doesn't exist
                os.makedirs(output_folder, exist_ok=True)
                
                # Change the file extension to .png
                base_name = os.path.splitext(file)[0]
                output_image_path = os.path.join(output_folder, f"{base_name}.png")
                
                print(f"Converting {input_image_path} to {output_image_path}")
                convert_32bit_to_16bit(input_image_path, output_image_path)

if __name__ == "__main__":
    # Specify the input root folder containing subfolders with .tif images
    input_root_folder = r"../../thermal_textures/tif_images/"

    # Specify the output root folder where .png images will be saved
    output_root_folder = r"../../thermal_textures/png_images/"

    convert_folder(input_root_folder, output_root_folder)
