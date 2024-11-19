# Dear Mr. Digruber,

# I would like to discuss with your progress on your project, as I have to
# evaluate it intermediately.

# Mohamed told me our technical part is working (i.e. renderer and
# simulator). The other student who generates the thermal textures can now
# also provide data.
# The thermal textures will be 512x512 in float32, having values in degree
# Celsius.

# In the Simulator, as far as I know, you need to set temperature values
# in Kelvin. So you might need to do some conversion here. In any case,
# simulated temperature values and temperatures in thermal textures must
# be in the right dimension.

# As you know, these are the steps that you need to do for generating one
# simulated integral image on your side:

# 1. Pick a thermal texture from the other student's database by random.

# 2. Compute a lable mask for that image. All pixels above sine
# temperature are labeled fire (1), all others are labeled no fire (0). I
# need to find out the temperature threshold with the fire fighters.

# 3. Setup the forest  with random parameters in the following ranges:
#      - numbers of tree per ha=  0...300 trees/ha
#      - ambient light =  0.5 ... 1
#      - azimuth angle of sunlight direction in degree = 0 ... 45
#      - compass direction of sunlight in degree = 0 ... 360
#      - tree top temperature in kelvin = XXX ... XXX (to be discussed -->
# this is in kelvin and not degrees, so you need to convert random values
# chosen in degrees C to kelvin, and also the thermal texture values must
# be converted to kelvin)
# To see how these parameters are set in the Simulator, please see the
# attached script. There are a couple of more parameters randomly chosen
# in that script that are related to person poses, etc. - you can ignore them.
# The size of the thermal textures on the ground is 35x35 meters - so your
# forest patch size must be 35x35m

# 3. Simulate 31 single (thermal) drone images for the forest above. Drone
# height is 35m AGL, all drone poses are simulated along a line (e.g., top
# to bottom), 0.5m spacing between each. The drone camera (again, you are
# simulating thermal in Gazebo) points straight down
# (photo_shoot_config.add_poses([gzm.Pose3d(x, y, 35, 0.0, 1.57, 0) ])) .
# You need to set the field-of-view of the thermal camera in Gazebo to
# match the field-of-view of the real drone's camera: horizontal/vertical
# FOV: 35° at image resolution of 512x512, the resolution of the images
# you render is 512x512 and they should be float32 with temperature values
# in degree C at each pixel (so they match exactly the images we get from
# the real drone - and the thermal texture images you used initially). You
# need to crop this to 512x512.

# 4. Compute the integral image with the AOS renderer for the 31 images in
# 3 Here, you need to chose the center drone (image 16/31) for the virtual
# camera. Pose parameters are the same as in 3/4, Focal plane is on the
# ground, and synthetic aperture is full up (all 31 images).

# That's it. This image pair (integral of temperatures from 4 and ground
# truth label image from 2) is what you need to store in your database.
# Both are 512x512 and spatially aligned. The first image is float 32 with
# temperatures in deg C, the other one is binary. You repeat steps 1-4
# often to get more images in the database. That will take some time, as
# the simulator is slow. So you need to start with it soon.

# So, I suggest a meeting on next Monday 3pm (I have a seminar at 5pm),
# where you present your status and we discuss possible questions
# regarding the above. Please set up a zoom-link with Mohamed and share it
# with me.


import sdformat13 as sdf
import gz.math7 as gzm

from photo_shoot_config import PhotoShootConfig
from person_config import PersonConfig
from forest_config import ForestConfig
from world_config import WorldConfig
from launcher import Launcher
import numpy as np
import random
import time
from PIL import Image
import math
import os

from typing import Tuple, Union
from math import sin, cos, atan2, sqrt

Number = Union[int, float]
Vector = Tuple[Number, Number, Number]


class SimulationRunner:
    def __init__(self):
        # Initialize file paths and directories
        self.world_file_in = "../../worlds/example_photo_shoot.sdf"
        self.world_file_out = "../../photo_shoot.sdf"
        self.output_directory = "../../data/simulator"

        # Load the initial world configuration
        self.world_config = WorldConfig()
        self.world_config.load(self.world_file_in)

        # Initialize parameter ranges and indices
        self.thermal_index = [4]

        # Initialize counters and iteration settings
        self.PC_Num = 0
        self.iter_Number = 1000000
        self.temperature_threshold= 500

    def run(self):
        for i in range(self.iter_Number):
            # Reload world_config for each iteration
            self.world_config = WorldConfig()
            self.world_config.load(self.world_file_in)

            # Generate random parameters
            self.generate_random_parameters()

            for j in range(2):  # Loop to generate GT with the same parameters
                # Configure the simulation components
                self.configure_light_and_scene()
                self.configure_photo_shoot(j, i)
                self.configure_forest(j)

                if j == 0:
                    self.write_parameters_file(i)
                    self.write_poses(i)

                # Save and launch the simulation
                self.save_world_config()
                self.launch_simulation()


            # Print iteration info
            print(f"\nIteration {i} / {self.iter_Number - 1} is Done\n")

    
    
    def generate_random_parameters(self):
        # Generate random value of number of trees
        time.sleep(1)
        random.seed(int(time.time()))
        self.x_rand_treeNum = random.randint(0, 300)
        print("x_rand_treeNum =", self.x_rand_treeNum)

        # Generate random thermal index
        time.sleep(1)
        random.seed(int(time.time()))
        self.rand_thermal = random.randint(0, len(self.thermal_index) - 1)
        print("rand_thermal =", self.rand_thermal)

        # Generate random value for ambient light
        time.sleep(1)
        random.seed(int(time.time()))
        self.x_rand_ambient = random.uniform(0.5, 1)
        print("x_rand_ambient =", self.x_rand_ambient)

        # Generate Alpha and Beta values for light direction
        time.sleep(1)
        random.seed(int(time.time()))
        self.x_rand_Alfa = random.randint(0, 45)
        print("x_rand_Alpha in degree =", self.x_rand_Alfa)
        self.x_rand_Alfa_rad = math.radians(self.x_rand_Alfa)

        time.sleep(1)
        random.seed(int(time.time()))
        self.x_rand_Beta = random.randint(0, 360)
        print("x_rand_Beta in degree =", self.x_rand_Beta)
        self.x_rand_Beta_rad = math.radians(self.x_rand_Beta)

        print("x_rand_Alfa in radian =", self.x_rand_Alfa_rad)
        print("x_rand_Beta in radian =", self.x_rand_Beta_rad)

        # Convert spherical coordinates to cartesian
        self.x_1, self.x_2, self.x_3 = self.to_cartesian(
            1, self.x_rand_Alfa_rad, self.x_rand_Beta_rad)
        if self.x_3 > 0:
            self.x_3 = self.x_3 * (-1)

        print("x_1 =", self.x_1)
        print("x_2 =", self.x_2)
        print("x_3 =", self.x_3)

        # Generate Ground Temperature
        time.sleep(1)
        random.seed(int(time.time()))
        self.x_rand_Groud = random.randint(260, 310)
        print("Ground Temperature =", self.x_rand_Groud)

        # Generate Tree Top Temperature
        time.sleep(1)
        random.seed(int(time.time()))
        self.x_rand_Tree = random.randint(295, 312)
        print("Tree Top Temperature =", self.x_rand_Tree)

    def configure_light_and_scene(self):
        # Configure the sun as the light source
        light = self.world_config.get_light("sun")
        light.set_direction(gzm.Vector3d(self.x_1, self.x_2, self.x_3))
        light.set_cast_shadows(False)

        # Configure the scene
        scene = self.world_config.get_scene()
        scene.set_ambient(gzm.Color(
            self.x_rand_ambient, self.x_rand_ambient, self.x_rand_ambient, 1.0))

    def configure_photo_shoot(self, j, i):
        
        photo_shoot_config = PhotoShootConfig()

        # Generate folder for every iteration
        self.patch_folder = os.path.join(self.output_directory, f"{i}")

        try:
            os.remove(self.patch_folder)
        except:
            pass


        if not os.path.exists(self.patch_folder):
            os.makedirs(self.patch_folder)
        

        photo_shoot_config.set_directory(self.patch_folder)

        if j == 0:
            img_Name = f"{self.PC_Num}_{i}"
        else:
            img_Name = f"{self.PC_Num}_{i}_GT"
        photo_shoot_config.set_prefix(img_Name)

        photo_shoot_config.set_direct_thermal_factor(64)
        photo_shoot_config.set_indirect_thermal_factor(5)

        photo_shoot_config.set_save_rgb(False)
        photo_shoot_config.set_save_thermal(True)
        photo_shoot_config.set_save_depth(False)

        photo_shoot_config.set_lower_thermal_threshold(285)
        photo_shoot_config.set_upper_thermal_threshold(530)

        if j == 0:
            self.flight_path = [x for x in range(30, -30, -1)]
            photo_shoot_config.add_poses([gzm.Pose3d(15, x, 35, 0.0, 1.57, 0)] for x in range(30,-30, -1))
        else:
            photo_shoot_config.add_poses([
                gzm.Pose3d(15, 0, 35, 0.0, 1.57, 0)
            ])

        self.world_config.add_plugin(photo_shoot_config)

        self.save_label_mask()

    def compute_label_mask(self):
        # Load the thermal image
        thermal_image = np.array(Image.open(self.thermal_texture_path))
        
        # Convert to Kelvin if necessary (assuming the image is in degrees Celsius)
        thermal_image_kelvin = thermal_image + 273.15
        
        # Create the label mask
        label_mask = np.where(thermal_image_kelvin > self.temperature_threshold, 1, 0)
        return label_mask

    def save_label_mask(self):
        self.thermal_texture_path = f"/home/mdigruber/gazebo_sim/models/procedural-forest/materials/textures/ground_00{self.thermal_index[self.rand_thermal]}_thermal.png"
        label_mask = self.compute_label_mask()
        
        label_mask_path = os.path.join(self.patch_folder, 'label_mask.npy')
        np.save(label_mask_path, label_mask)

        label_mask_image = Image.fromarray((label_mask * 255).astype('uint8'))
        label_mask_path = os.path.join(self.patch_folder, 'label_mask.bmp')
        label_mask_image.save(label_mask_path, format='BMP')

    
    def configure_forest(self, j):
        self.forest_config = ForestConfig()
        forest_config = self.forest_config

        forest_config.set_generate(True)
        forest_config.set_ground_texture(0)
        forest_config.set_direct_spawning(True)
        forest_config.set_texture_size(35)

        forest_config.set_ground_thermal_texture(
            self.thermal_index[self.rand_thermal],
            288.15,  # Minimal temperature in Kelvin
            520.0    # Maximal temperature in Kelvin
        )
        #forest_config.set_trunk_temperature(290)
        forest_config.set_twigs_temperature(self.x_rand_Tree)
        forest_config.set_size(35)

        if j == 0:
            forest_config.set_trees(self.x_rand_treeNum)
        else:
            forest_config.set_trees(0)

        time.sleep(1)
        forest_config.set_seed(int(time.time()))
        forest_config.set_species("Birch", {
            "percentage": 1.0,
            "homogeneity": 0.95,
            "trunk_texture": 0,
            "twigs_texture": 0,
            "tree_properties": {
                "clump_max": 0.45,
                "clump_min": 0.4,
                "length_falloff_factor": 0.65,
                "length_falloff_power": 0.75,
                "branch_factor": 2.45,
                "radius_falloff_rate": 0.7,
                "climb_rate": 0.55,
                "taper_rate": 0.8,
                "twist_rate": 8.0,
                "segments": 6,
                "levels": 6,
                "sweep_amount": 0.0,
                "initial_branch_length": 0.7,
                "trunk_length": 1.0,
                "drop_amount": 0.0,
                "grow_amount": 0.4,
                "v_multiplier": 0.2,
                "twig_scale": 0.2
            }
        })

        self.world_config.add_plugin(forest_config)

    def write_parameters_file(self, i):
        label_path = f"{self.patch_folder}/parameters.txt"

        with open(label_path, "w+") as file:
            file.write(f"img_GT {(0, 0, 35, 0.0, 1.57, 0)}\n")
            file.write('\n')
            for coords in self.flight_path:
                file.write(f"img_1 {(coords, 0, 35, 0.0, 1.57, 0)}\n")
            file.write('\n')
            file.write(
                f"numbers of tree per ha= {self.forest_config.config['trees']}\n")
            file.write('\n')
            file.write(f"ambient light = {self.x_rand_ambient}\n")
            file.write('\n')
            file.write(
                f"azimuth angle of sun light in degrees = {self.x_rand_Alfa}\n")
            file.write(
                f"compass direction of sunlight in degrees = {self.x_rand_Beta}\n")
            file.write('\n')
            file.write(
                f"ground surface temperature in kelvin = {self.x_rand_Groud}\n")
            file.write(
                f"tree top temperature in kelvin = {self.x_rand_Tree}\n")

    def write_poses(self, i):
        label_path = f"{self.patch_folder}/poses.txt"

        with open(label_path, "w+") as file:
            for coords in self.flight_path:
                file.write(f"15, {coords}, 35\n")
            

    def save_world_config(self):
        self.world_config.save(self.world_file_out)

    def launch_simulation(self):
        launcher = Launcher()
        launcher.set_launch_config("server_only", True)
        launcher.set_launch_config("running", True)
        launcher.set_launch_config("iterations", 2)
        launcher.set_launch_config("world", self.world_file_out)
        print(launcher.launch())

    @staticmethod
    def distance(a: Vector, b: Vector) -> Number:
        """Returns the distance between two cartesian points."""
        x = (b[0] - a[0]) ** 2
        y = (b[1] - a[1]) ** 2
        z = (b[2] - a[2]) ** 2
        return (x + y + z) ** 0.5

    @staticmethod
    def magnitude(x: Number, y: Number, z: Number) -> Number:
        """Returns the magnitude of the vector."""
        return sqrt(x * x + y * y + z * z)

    @staticmethod
    def to_spherical(x: Number, y: Number, z: Number) -> Vector:
        """Converts a cartesian coordinate (x, y, z) into a spherical one (radius, theta, phi)."""
        radius = SimulationRunner.magnitude(x, y, z)
        theta = atan2(sqrt(x * x + y * y), z)
        phi = atan2(y, x)
        return (radius, theta, phi)

    @staticmethod
    def to_cartesian(radius: Number, theta: Number, phi: Number) -> Vector:
        """Converts a spherical coordinate (radius, theta, phi) into a cartesian one (x, y, z)."""
        x = radius * cos(phi) * sin(theta)
        y = radius * sin(phi) * sin(theta)
        z = radius * cos(theta)
        return (x, y, z)


if __name__ == "__main__":
    simulation_runner = SimulationRunner()
    simulation_runner.run()
