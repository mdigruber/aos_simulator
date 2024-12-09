import numpy as np
from PIL import Image


og_image = np.array(Image.open("./000017.TIF"))
texture = np.array(Image.open("./output/000017.png"))
gt_image = np.array(Image.open("./output/gt_0_0_pose_0_thermal.png"))
integral_image = np.array(Image.open("./output/integral_0.png"))

def get_deg_c(value):
	minK = og_image.min() + 273.15
	maxK = og_image.max() + 273.15
	return (value/255*(maxK - minK)+minK)-273.15

print("Original Texture")
print(og_image.min())
print(og_image.max())
print("\nGenerated Texture")
print(get_deg_c(texture.min()))
print(get_deg_c(texture.max()))
print("\nGround Truth")
print(get_deg_c(gt_image.min()))
print(get_deg_c(gt_image.max()))
print("\nIntegral Image")
print(get_deg_c(integral_image.min()))
print(get_deg_c(integral_image.max()))




