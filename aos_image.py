### Make sure the file is executed from the LFR/python directory

## Import libraries section ##
import numpy as np
import cv2
import os
import math
import re
import LFR_utils as utils
import pyaos
import glm
import glob
import re
import tifffile


image_folder_path = r"/home/mdigruber/AOS/AOS for Drone Swarms/LFR/data/img_markus"  # Enter path to where your images are saved.
results_path = r"/home/mdigruber/AOS/AOS for Drone Swarms/LFR/data/FP"  # Enter path to where you want the results to be saved.
set_folder = r"/home/mdigruber/AOS/AOS for Drone Swarms/LFR/python"  # Enter path to your LFR/python directory
pose_file = r"/home/mdigruber/AOS/AOS for Drone Swarms/LFR/data/poses.txt"  # Enter path to your LFR/python directory

w, h, fovDegrees = 512, 512, 35
render_fov = 35

number_of_images = 31
focal_plane = 34

if "window" not in locals() or window == None:
    window = pyaos.PyGlfwWindow(w, h, "AOS")

aos = pyaos.PyAOS(w, h, fovDegrees)

aos.loadDEM(os.path.join(set_folder, "zero_plane.obj"))


def eul2rotm(theta):
    s_1 = math.sin(theta[0])
    c_1 = math.cos(theta[0])
    s_2 = math.sin(theta[1])
    c_2 = math.cos(theta[1])
    s_3 = math.sin(theta[2])
    c_3 = math.cos(theta[2])
    rotm = np.identity(3)
    rotm[0, 0] = c_1 * c_2
    rotm[0, 1] = c_1 * s_2 * s_3 - s_1 * c_3
    rotm[0, 2] = c_1 * s_2 * c_3 + s_1 * s_3

    rotm[1, 0] = s_1 * c_2
    rotm[1, 1] = s_1 * s_2 * s_3 + c_1 * c_3
    rotm[1, 2] = s_1 * s_2 * c_3 - c_1 * s_3

    rotm[2, 0] = -s_2
    rotm[2, 1] = c_2 * s_3
    rotm[2, 2] = c_2 * c_3

    return rotm


def createviewmateuler(eulerang, camLocation):
    rotationmat = eul2rotm(eulerang)
    translVec = np.reshape((-camLocation @ rotationmat), (3, 1))
    conjoinedmat = np.append(np.transpose(rotationmat), translVec, axis=1)
    return conjoinedmat


def divide_by_alpha(rimg2):
    a = np.stack((rimg2[:, :, 3], rimg2[:, :, 3], rimg2[:, :, 3]), axis=-1)
    return rimg2[:, :, :3] / a


def pose_to_virtualcamera(vpose):
    vp = glm.mat4(*np.array(vpose).transpose().flatten())
    ivp = glm.inverse(glm.transpose(vp))
    Posvec = glm.vec3(ivp[3])
    Upvec = glm.vec3(ivp[1])
    FrontVec = glm.vec3(ivp[2])
    lookAt = glm.lookAt(Posvec, Posvec + FrontVec, Upvec)
    cameraviewarr = np.asarray(lookAt)
    return cameraviewarr


def utils_pose_to_virtualcamera(vpose):
    vp = glm.mat4(*np.array(vpose).transpose().flatten())
    ivp = glm.inverse(glm.transpose(vp))
    Posvec = glm.vec3(ivp[3])
    Upvec = glm.vec3(ivp[1])
    FrontVec = glm.vec3(ivp[2])
    lookAt = glm.lookAt(Posvec, Posvec + FrontVec, Upvec)
    return np.asarray(glm.transpose(lookAt))

with open(pose_file, "r") as f:
    lines = f.readlines()

image_posx = []
image_posy = []
image_posz = []


for line in lines:
    image_posx.append(float(line.split(",")[0]))
    image_posy.append(float(line.split(",")[1]) * -1)
    image_posz.append(float(line.split(",")[2]))

ref_loc = [image_posx, image_posy]
altitude_list = image_posz

center_index = number_of_images // 2 + 1

print(center_index)

site_poses = []
for i in range(number_of_images):
    M = createviewmateuler(
        np.array([0.0, 0.0, 0.0]),
        np.array([ref_loc[1][i], ref_loc[0][i], -altitude_list[i]]),
    )
    ViewMatrix = np.vstack((M, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)))
    camerapose = np.asarray(ViewMatrix.transpose(), dtype=np.float32)
    site_poses.append(
        camerapose
    )  # site_poses is a list now containing all the poses of all the images in a certain format that is accecpted by the renderer.


numbers = re.compile(r"(\d+)")


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


imagelist = []


for img in sorted(glob.glob(image_folder_path + "/*.png"), key=numericalSort):
    n = cv2.imread(img)
    imagelist.append(n)


aos.clearViews()  # Every time you call the renderer you should use this line to clear the previous views
for i in range(len(imagelist)):
    aos.addView(
        imagelist[i], site_poses[i], ""
    )  # Here we are adding images to the renderer one by one.

focal = 0

aos.setDEMTransform([0, 0, focal])
proj_RGBimg = aos.render(
    utils_pose_to_virtualcamera(site_poses[center_index]), render_fov
)

tmp_RGB = divide_by_alpha(proj_RGBimg)

cv2.imwrite(
    os.path.join(results_path, "integral.png"), tmp_RGB
)
