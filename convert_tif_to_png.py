import os
import numpy as np
from PIL import Image
from tifffile import imwrite

def convert_32bit_to_16bit(image_path, output_path):
    """
    Converts a 32-bit image to 16-bit, applies transformations, and saves as a PNG.
    
    Parameters:
    - image_path: Path to the input 32-bit image.
    - output_path: Path where the converted 16-bit PNG will be saved.
    """
    try:
        # Load the 32-bit image as a NumPy array
        image = Image.open(image_path)
        image_data = (np.array(image, dtype=np.float32) + 273.15) * 100  # Scale to 0 - 65535

        # Normalize the 32-bit image to the range [0, 65535]
        min_val = np.min(image_data)
        max_val = np.max(image_data)

        min_index = np.argmin(image_data)
        max_index = np.argmax(image_data)
       
        # Convert flattened indices to 2D indices
        min_index = np.unravel_index(min_index, image_data.shape)
        max_index = np.unravel_index(max_index, image_data.shape)

        print(f"Processing '{os.path.basename(image_path)}'")
        print(f"Min index: {min_index}, Max index: {max_index}")
        print(f"Min value: {image_data[min_index]}, Max value: {image_data[max_index]}")
        
        # Avoid division by zero in case of uniform images
        if max_val - min_val > 0:
            normalized_data = (image_data - min_val) / (max_val - min_val) * 65535
            env_temp = ((np.float32(10) - min_val) / (max_val - min_val) * 65535) * 0.01
        else:
            normalized_data = np.zeros_like(image_data)  # Handle uniform case

        # Convert to 16-bit integers
        image_16bit = normalized_data.astype(np.uint16)
        print(f"Min 16-bit value: {image_16bit[min_index]}, Max 16-bit value: {image_16bit[max_index]}")
        
        # Rotate the image 90 degrees to the right (clockwise)
        image_rotated = np.rot90(image_16bit, k=-1)  # k=-1 for clockwise rotation

        # Mirror the image horizontally
        image_transformed = np.fliplr(image_rotated)

        # Save the transformed image as PNG
        imwrite(output_path, image_transformed)
        print(f"Saved converted image to '{output_path}'\n")
    
    except Exception as e:
        print(f"Failed to process '{image_path}'. Error: {e}\n")

def convert_folder(input_folder, output_folder):
    """
    Converts all supported 32-bit images in the input folder to 16-bit PNGs in the output folder.
    
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
                
                convert_32bit_to_16bit(input_path, output_path)
            else:
                print(f"Skipping '{filename}': Unsupported file extension.\n")

if __name__ == "__main__":
    input_folder = r"/home/mdigruber/gazebo_simulator/models/procedural-forest/materials/textures/thermal"
    output_folder = r"/home/mdigruber/gazebo_simulator/models/procedural-forest/materials/textures/thermal/output"
    
    convert_folder(input_folder, output_folder)
