
### Make sure the file is executed from the LFR/python directory

## Import libraries section ##
import numpy as np
import cv2
import os
import math
import re
from LFR_utils import pose_to_virtualcamera
import LFR_utils as utils
import pyaos
import glm
import glob
import re


image_folder_path = r'/home/mdigruber/AOS/AOS for Drone Swarms/LFR/data/img_markus'     # Enter path to where your images are saved.
results_path = r'/home/mdigruber/AOS/AOS for Drone Swarms/LFR/data/FP'                  # Enter path to where you want the results to be saved.
set_folder = r'/home/mdigruber/AOS/AOS for Drone Swarms/LFR/python'                     # Enter path to your LFR/python directory

w,h,fovDegrees = 512, 512, 50
render_fov = 50

if 'window' not in locals() or window == None:
                                    
    window = pyaos.PyGlfwWindow( w, h, 'AOS' )  
     
aos = pyaos.PyAOS(w,h,fovDegrees) 

aos.loadDEM( os.path.join(set_folder,'zero_plane.obj'))

def eul2rotm(theta) :
    s_1 = math.sin(theta[0])
    c_1 = math.cos(theta[0]) 
    s_2 = math.sin(theta[1]) 
    c_2 = math.cos(theta[1]) 
    s_3 = math.sin(theta[2]) 
    c_3 = math.cos(theta[2])
    rotm = np.identity(3)
    rotm[0,0] =  c_1*c_2
    rotm[0,1] =  c_1*s_2*s_3 - s_1*c_3
    rotm[0,2] =  c_1*s_2*c_3 + s_1*s_3

    rotm[1,0] =  s_1*c_2
    rotm[1,1] =  s_1*s_2*s_3 + c_1*c_3
    rotm[1,2] =  s_1*s_2*c_3 - c_1*s_3

    rotm[2,0] = -s_2
    rotm[2,1] =  c_2*s_3
    rotm[2,2] =  c_2*c_3        

    return rotm

def createviewmateuler(eulerang, camLocation):
    
    rotationmat = eul2rotm(eulerang)
    translVec =  np.reshape((-camLocation @ rotationmat),(3,1))
    conjoinedmat = (np.append(np.transpose(rotationmat), translVec, axis=1))
    return conjoinedmat

def divide_by_alpha(rimg2):
        a = np.stack((rimg2[:,:,3],rimg2[:,:,3],rimg2[:,:,3]),axis=-1)
        return rimg2[:,:,:3]/a

def pose_to_virtualcamera(vpose ):
    vp = glm.mat4(*np.array(vpose).transpose().flatten())
    ivp = glm.inverse(glm.transpose(vp))
    Posvec = glm.vec3(ivp[3])
    Upvec = glm.vec3(ivp[1])
    FrontVec = glm.vec3(ivp[2])
    lookAt = glm.lookAt(Posvec, Posvec + FrontVec, Upvec)
    cameraviewarr = np.asarray(lookAt)
    print(cameraviewarr)
    return cameraviewarr  


number_of_images = 31
focal_plane = 0

ref_loc = [[-7.5, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],[0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0]]   # These are the x and y positions of the images. It is of the form [[x_positions],[y_positions]]

altitude_list = [35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35] # [Z values which is the height]

center_index = 5

site_poses = []
for i in range(number_of_images):
    EastCentered = (ref_loc[0][i] - 0.0) #Get MeanEast and Set MeanEast
    NorthCentered = (0.0 - ref_loc[1][i]) #Get MeanNorth and Set MeanNorth
    M = createviewmateuler(np.array([0.0, 0.0, 0.0]),np.array( [ref_loc[0][i], ref_loc[1][i], - altitude_list[i]] ))
    print('m',M)
    ViewMatrix = np.vstack((M, np.array([0.0,0.0,0.0,1.0],dtype=np.float32)))
    print(ViewMatrix)
    camerapose = np.asarray(ViewMatrix.transpose(),dtype=np.float32)
    print(camerapose)
    site_poses.append(camerapose)  # site_poses is a list now containing all the poses of all the images in a certain format that is accecpted by the renderer.


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

imagelist = []

for img in sorted(glob.glob(image_folder_path + '/*.png'),key=numericalSort):
    n= cv2.imread(img)
    imagelist.append(n)


aos.clearViews()   # Every time you call the renderer you should use this line to clear the previous views  
for i in range(len(imagelist)):
        aos.addView(imagelist[i], site_poses[i], "DEM BlobTrack")  # Here we are adding images to the renderer one by one.
aos.setDEMTransform([0,0,focal_plane])


proj_RGBimg = aos.render(utils.pose_to_virtualcamera(site_poses[center_index]), render_fov)
tmp_RGB = divide_by_alpha(proj_RGBimg)
cv2.imwrite(os.path.join( results_path, 'integral.png'), tmp_RGB) 
