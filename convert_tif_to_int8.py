import os
import numpy as np
from PIL import Image
from tifffile import imwrite
import cv2  # Ensure OpenCV is installed: pip install opencv-python

def convert_32bit_to_8bit(image_path, output_path):
    """
    Converts a 32-bit float TIF image to an 8-bit PNG, applying transformations and thermal scaling.

    Parameters:
    - image_path: Path to the input 32-bit TIF image.
    - output_path: Path where the converted 8-bit PNG will be saved.
    """
    try:
        # Load the 32-bit image as a NumPy array
        with Image.open(image_path) as img:
            image_data = np.array(img, dtype=np.float32)

        # Debug: Print shape and data type
        print(f"Loaded image '{os.path.basename(image_path)}' with shape {image_data.shape} and dtype {image_data.dtype}")

        # Convert Celsius to Kelvin
        image_data_kelvin = image_data + 273.15

        # Calculate dynamic min and max from the image data
        minK = np.min(image_data_kelvin)
        maxK = np.max(image_data_kelvin)

        # Handle cases where minK == maxK to avoid division by zero
        if minK == maxK:
            print(f"Image '{os.path.basename(image_path)}' has uniform temperature. Creating a zeroed 8-bit image.")
            image_8bit = np.zeros_like(image_data_kelvin, dtype=np.uint8)
        else:
            # Scale the temperature data to [0, 255]
            normalized_data = (image_data_kelvin - minK) / (maxK - minK) * 255.0
            normalized_data = np.clip(normalized_data, 0, 255)  # Ensure values are within [0, 255]

            # Convert to 8-bit unsigned integers
            image_8bit = normalized_data.astype(np.uint8)

            # Debug: Check min and max after normalization
            print(f"After normalization: Min value = {image_8bit.min()}, Max value = {image_8bit.max()}")

        # Rotate the image 90 degrees to the right (clockwise)
        image_rotated = np.rot90(image_8bit, k=-1)  # k=-1 for clockwise rotation

        # Mirror the image horizontally
        image_transformed = np.fliplr(image_rotated)

        # Save the transformed image as PNG
        imwrite(output_path, image_transformed)
        print(f"Saved converted 8-bit image to '{output_path}'\n")

    except Exception as e:
        print(f"Failed to process '{image_path}'. Error: {e}\n")

def convert_folder(input_folder, output_folder):
    """
    Converts all supported 32-bit images in the input folder to 8-bit PNGs in the output folder.

    Parameters:
    - input_folder: Directory containing the input images.
    - output_folder: Directory where converted images will be saved.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    supported_extensions = ['.tif', '.tiff']

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        # Construct full file path
        input_path = os.path.join(input_folder, filename)

        # Check if it's a file and has a supported extension
        if os.path.isfile(input_path):
            _, ext = os.path.splitext(filename)
            if ext.lower() in supported_extensions:
                # Define the output path with .png extension
                output_filename = os.path.splitext(filename)[0] + ".png"
                output_path = os.path.join(output_folder, output_filename)
                
                convert_32bit_to_8bit(input_path, output_path)
            else:
                print(f"Skipping '{filename}': Unsupported file extension.\n")

if __name__ == "__main__":
    input_folder = r"/home/mdigruber/gazebo_simulator/models/procedural-forest/materials/textures/thermal"
    output_folder = r"/home/mdigruber/gazebo_simulator/models/procedural-forest/materials/textures/thermal/output"

    convert_folder(input_folder, output_folder)
