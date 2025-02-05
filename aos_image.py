import os
import math
import re
import glob

import cv2
import numpy as np
import glm
import pyaos

class AOSRenderer:
 
    def __init__(
        self,
        image_folder_path,
        results_path,
        set_folder,
        pose_file,
        w=512,
        h=512,
        fovDegrees=35,
        render_fov=35,
        number_of_images=31,
        focal_plane=34,
    ):
        self.image_folder_path = image_folder_path
        self.results_path = results_path
        self.set_folder = set_folder
        self.pose_file = pose_file
        self.w = w
        self.h = h
        self.fovDegrees = fovDegrees
        self.render_fov = render_fov
        self.number_of_images = number_of_images
        self.focal_plane = focal_plane

    
        os.makedirs(self.results_path, exist_ok=True)

        self.window = pyaos.PyGlfwWindow(self.w, self.h, "AOS")
    
        self.aos = pyaos.PyAOS(self.w, self.h, self.fovDegrees)
    
        dem_path = os.path.join(self.set_folder, "zero_plane.obj")
        self.aos.loadDEM(dem_path)

        self.imagelist = []
        self.site_poses = []
        self.center_index = None

    def _eul2rotm(self, theta):
        """Convert Euler angles to a rotation matrix."""
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

    def _createviewmateuler(self, eulerang, camLocation):
        
        rotationmat = self._eul2rotm(eulerang)
        translVec = np.reshape((-camLocation @ rotationmat), (3, 1))
        conjoinedmat = np.append(np.transpose(rotationmat), translVec, axis=1)
        return conjoinedmat

    def _divide_by_alpha(self, rimg2):
        a = np.stack((rimg2[:, :, 3], rimg2[:, :, 3], rimg2[:, :, 3]), axis=-1)
        return rimg2[:, :, :3] / a

    def _utils_pose_to_virtualcamera(self, vpose):
        vp = glm.mat4(*np.array(vpose).transpose().flatten())
        ivp = glm.inverse(glm.transpose(vp))
        Posvec = glm.vec3(ivp[3])
        Upvec = glm.vec3(ivp[1])
        FrontVec = glm.vec3(ivp[2])
        lookAt = glm.lookAt(Posvec, Posvec + FrontVec, Upvec)
        return np.asarray(glm.transpose(lookAt))

    @staticmethod
    def _numericalSort(value):
        """Sort key function to sort filenames containing numbers."""
        numbers = re.compile(r"(\d+)")
        parts = numbers.split(value)
        parts[1::2] = list(map(int, parts[1::2]))
        return parts

    def _load_poses(self):
        """Read the pose file and compute the camera poses for each image."""
        with open(self.pose_file, "r") as f:
            lines = f.readlines()

        image_posx = []
        image_posy = []
        image_posz = []


        for line in lines:
            parts = line.strip().split(",")
            image_posx.append(float(parts[0]))
            image_posy.append(float(parts[1]) * -1)
            image_posz.append(float(parts[2]))


        ref_loc = [image_posx, image_posy]
        altitude_list = image_posz

        self.center_index = self.number_of_images // 2 + 1
        print("Center index:", self.center_index)

        # Build the list of site poses
        self.site_poses = []
        for i in range(self.number_of_images):
            M = self._createviewmateuler(
                np.array([0.0, 0.0, 0.0]),
                np.array([ref_loc[1][i], ref_loc[0][i], -altitude_list[i]]),
            )
            ViewMatrix = np.vstack(
                (M, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            )
            camerapose = np.asarray(ViewMatrix.transpose(), dtype=np.float32)
            self.site_poses.append(camerapose)

    def _load_images(self):
        """Load and sort the image files from the image folder."""
        file_pattern = os.path.join(self.image_folder_path, "*.png")
        files = sorted(glob.glob(file_pattern), key=self._numericalSort)
        self.imagelist = []
        for img_path in files:
            img = cv2.imread(img_path)
            if img is not None:
                self.imagelist.append(img)
                
        # Use only the first number_of_images images
        self.imagelist = self.imagelist[: self.number_of_images]

    def _setup_views(self):
        """Clear previous views and add new views to the renderer."""
        self.aos.clearViews()
        for i in range(len(self.imagelist)):
            if i < len(self.site_poses):
                self.aos.addView(self.imagelist[i], self.site_poses[i], "")
            else:
                print(f"Warning: No pose available for image index {i}.")

    def render(self):
        self._load_poses()
        self._load_images()

        self._setup_views()
        focal = 0
        self.aos.setDEMTransform([0, 0, focal])

        self.aos.clearViews()

        if self.center_index is None or self.center_index >= len(self.site_poses):
            print("Error: center_index out of range, using index 0 instead.")
            cam_pose = self.site_poses[0]
        else:
            cam_pose = self.site_poses[self.center_index]
    
    
        proj_RGBimg = self.aos.render(
            self._utils_pose_to_virtualcamera(cam_pose), self.render_fov
        )

        tmp_RGB = self._divide_by_alpha(proj_RGBimg)

        output_filename = "integral.png"
        output_path = os.path.join(self.results_path, output_filename)
        cv2.imwrite(output_path, tmp_RGB)
        print("Rendered image saved to", output_path)

        return tmp_RGB