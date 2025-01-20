import os
import random
import numpy as np
from PIL import Image
import math
import shutil
import gz.math7 as gzm
from photo_shoot_config import PhotoShootConfig
from forest_config import ForestConfig
from world_config import WorldConfig
from launcher import Launcher
from typing import Tuple, Union
from math import sin, cos, atan2, sqrt


Number = Union[int, float]
Vector = Tuple[Number, Number, Number]


class SimulationRunner:
    def __init__(self):
        
        # Initialize file paths and directories
        self.world_file_in = "../../worlds/example_photo_shoot.sdf"
        self.world_file_out = "../../photo_shoot.sdf"
        self.output_directory = "/home/mdigruber/Desktop/simulator"
        self.database_dir = "/home/mdigruber/training_data/"

        # Load the initial world configuration
        self.world_config = WorldConfig()
        self.world_config.load(self.world_file_in)

        # Initialize counters and iteration settings
        self.PC_Num = 0
        self.iter_Number = 1000000
        self.iteration = 0
        
        # Set a default temperature threshold (in degrees Celsius)
        self.temperature_threshold_C = 25 # somewhere between 25 and 60 
        self.temperature_threshold_K = self.temperature_threshold_C + 273.15

        # Path to the thermal texture database
        self.thermal_texture_dir = "/home/mdigruber/gazebo_simulator/models/procedural-forest/materials/textures/thermal/output"

    def run(self):
        for i in range(self.iter_Number):
            self.iteration = i
            # Reload world_config for each iteration
            self.world_config = WorldConfig()
            self.world_config.load(self.world_file_in)

            # Generate random parameters
            self.generate_random_parameters()

            # Configure the simulation components
            self.configure_light_and_scene()
            self.configure_photo_shoot(i)

            # Config for Forest
            self.configure_forest()

            # Save the world configuration
            self.save_world_config()

            # Launch the simulation
            self.launch_simulation()

            # Generate ground truth image
            self.generate_ground_truth(i)

            # Print iteration info
            print(f"\nIteration {i + 1} / {self.iter_Number} is running\n")

    def generate_random_parameters(self):
        
        # Randomly select a thermal texture
        thermal_textures = [
            f for f in os.listdir(self.thermal_texture_dir) if f.endswith(".png")
        ]
        self.thermal_texture = random.choice(thermal_textures)
        print(f"Selected thermal texture: {self.thermal_texture}")

        # Number of trees per hectare (ha)
        self.x_rand_treeNum = random.randint(0, 300)
        print("Number of trees per hectare =", self.x_rand_treeNum)

        # Ambient light
        self.x_rand_ambient = random.uniform(0.5, 1.0)

        print("Ambient light =", self.x_rand_ambient)
        
        # Azimuth angle of sunlight direction (Alpha)
        self.x_rand_Alpha = random.uniform(0, 45)
        self.x_rand_Alpha_rad = math.radians(self.x_rand_Alpha)
        print("Azimuth angle (Alpha) in degrees =", self.x_rand_Alpha)

        # Compass direction of sunlight (Beta)
        self.x_rand_Beta = random.uniform(0, 360)
        self.x_rand_Beta_rad = math.radians(self.x_rand_Beta)
        print("Compass direction (Beta) in degrees =", self.x_rand_Beta)

        # Convert spherical coordinates to Cartesian for light direction
        self.x_1, self.x_2, self.x_3 = self.to_cartesian(
            1, self.x_rand_Alpha_rad, self.x_rand_Beta_rad
        )
        if self.x_3 > 0:
            self.x_3 = -self.x_3  # Ensure sunlight comes from above

        # Tree top temperature in degrees Celsius and convert to Kelvin
        self.x_rand_Tree_C = random.uniform(15, 30)
        self.x_rand_Tree = self.x_rand_Tree_C + 273.15
        print(f"Tree top temperature: {self.x_rand_Tree_C}°C / {self.x_rand_Tree}K")

    def configure_light_and_scene(self):
        # Configure the sun as the light source
        light = self.world_config.get_light("sun")
        light.set_direction(gzm.Vector3d(self.x_1, self.x_2, self.x_3))
        light.set_cast_shadows(False)

        # Configure the scene
        scene = self.world_config.get_scene()
        scene.set_ambient(
            gzm.Color(
                self.x_rand_ambient, self.x_rand_ambient, self.x_rand_ambient, 1.0
            )
        )

    def configure_photo_shoot(self, i: int):
        photo_shoot_config = PhotoShootConfig()

        # Generate folder for every iteration
        self.patch_folder = os.path.join(self.output_directory, f"{i}")

        # Remove the folder if it exists
        if os.path.exists(self.patch_folder):
            shutil.rmtree(self.patch_folder)

        os.makedirs(self.patch_folder)

        # Gets the min and max temperature in Kelvin

        photo_shoot_config.set_directory(self.patch_folder)

        img_Name = f"{self.PC_Num}_{i}"
        photo_shoot_config.set_prefix(img_Name)

        # Set camera properties
        photo_shoot_config.set_direct_thermal_factor(20)  # direct sunlight
        photo_shoot_config.set_indirect_thermal_factor(5)  # indirect sunlight

        photo_shoot_config.set_save_rgb(False)
        photo_shoot_config.set_save_thermal(True)
        photo_shoot_config.set_save_depth(False)

        # Saves label Mask
        self.save_label_mask()

        # Set thermal thresholds based on expected temperature ranges
        lower_thermal_threshold = self.min_ground_temp_K - 10
        upper_thermal_threshold = self.max_ground_temp_K + 10

        photo_shoot_config.set_lower_thermal_threshold(lower_thermal_threshold)
        photo_shoot_config.set_upper_thermal_threshold(upper_thermal_threshold)

        # Define drone poses along a straight line with 0.5m spacing (vertical)
        num_images = 31
        spacing = 0.5

        inverse_x = True

        if inverse_x:
            self.x_positions = [
                (i * spacing - ((num_images - 1) / 2) * spacing) * -1
                for i in range(num_images)
            ]
        else:
            self.x_positions = [
                i * spacing - ((num_images - 1) / 2) * spacing
                for i in range(num_images)
            ]

        self.poses = [
            gzm.Pose3d(0, x, 35, 0.0, 1.57, 0.0) for x in self.x_positions
        ]  # x, y, z, -rotation, tilt angle, +rotation

        photo_shoot_config.add_poses(self.poses)

        self.world_config.add_plugin(photo_shoot_config)
        self.write_poses()

    def configure_forest(self):
        self.forest_config = ForestConfig()
        forest_config = self.forest_config

        forest_config.set_generate(True)
        forest_config.set_ground_texture(0)
        forest_config.set_direct_spawning(True)
        forest_config.set_texture_size(35)

        # Use the selected thermal texture
        forest_config.set_ground_thermal_texture(
            os.path.join(self.thermal_texture_dir, self.thermal_texture),
            self.min_ground_temp_K,  # Minimal temperature in Kelvin
            self.max_ground_temp_K,  # Maximal temperature in Kelvin
        )

        forest_config.set_trunk_temperature(self.min_ground_temp_K)  # In Kelvin
        forest_config.set_twigs_temperature(self.min_ground_temp_K)  # In Kelvin
        forest_config.set_size(100)  # Set forest size to 35x35 meters

        forest_config.set_trees(self.x_rand_treeNum)

        # Define tree species and properties (adjust as needed)
        forest_config.set_species(
            "Birch",
            {
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
                    "twig_scale": 0.2,
                },
            },
        )

        self.world_config.add_plugin(forest_config)

    def compute_label_mask(self):
        thermal_texture = "/home/mdigruber/gazebo_simulator/models/procedural-forest/materials/textures/thermal"
        thermal_texture_tif_image = os.path.join(
            thermal_texture, self.thermal_texture.replace(".png", ".TIF")
        )
        thermal_image = np.array(Image.open(thermal_texture_tif_image))

        thermal_image_K = thermal_image + 273.15

        self.min_ground_temp_K = thermal_image_K.min()
        self.max_ground_temp_K = thermal_image_K.max()

        print(f"Min Ground Temp: {self.min_ground_temp_K - 273.15} C / {self.min_ground_temp_K} K")
        print(f"Max Ground Temp: {self.max_ground_temp_K - 273.15} C / {self.max_ground_temp_K} K")

        label_mask = np.where(thermal_image >= self.temperature_threshold_C, 1, 0)

        return label_mask

    def save_label_mask(self):
        label_mask = self.compute_label_mask()

        label_mask_path = os.path.join(self.patch_folder, "label_mask.npy")
        np.save(label_mask_path, label_mask)

    def save_world_config(self):
        self.world_config.save(self.world_file_out)

    def launch_simulation(self):
        launcher = Launcher()
        launcher.set_launch_config("server_only", True)
        launcher.set_launch_config("running", True)
        launcher.set_launch_config("iterations", 2)
        launcher.set_launch_config("world", self.world_file_out)
        print(launcher.launch())

    def compute_integral_image(self):
        pass

    def write_poses(self):
        label_path = f"{self.patch_folder}/poses.txt"

        with open(label_path, "w+") as file:
            for coords in self.x_positions:
                file.write(f"0,{coords},35\n")

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
        x = radius * sin(theta) * cos(phi)
        y = radius * sin(theta) * sin(phi)
        z = radius * cos(theta)
        return (x, y, z)

    def generate_ground_truth(self, i):
        # Reload world_config
        self.world_config = WorldConfig()
        self.world_config.load(self.world_file_in)

        # Use the same random parameters generated before
        # Configure light and scene with the same parameters
        self.configure_light_and_scene()

        # Configure photo shoot with only one pose at (0,0,35)
        self.configure_photo_shoot_ground_truth(i)

        # Remove forest by setting trees to zero
        self.configure_forest_ground_truth()

        # Save the world configuration
        self.save_world_config()

        # Launch the simulation
        self.launch_simulation()

    def configure_photo_shoot_ground_truth(self, i):
        photo_shoot_config = PhotoShootConfig()

        # Use the same folder
        photo_shoot_config.set_directory(self.patch_folder)

        # Set a different prefix for the ground truth images
        img_Name = f"gt_{self.PC_Num}_{i}"
        photo_shoot_config.set_prefix(img_Name)

        # Set camera properties as before
        photo_shoot_config.set_direct_thermal_factor(20)  # direct sunlight
        photo_shoot_config.set_indirect_thermal_factor(5)  # indirect sunlight

        photo_shoot_config.set_save_rgb(False)
        photo_shoot_config.set_save_thermal(True)
        photo_shoot_config.set_save_depth(False)

        # Set thermal thresholds based on expected temperature ranges
        lower_thermal_threshold = self.min_ground_temp_K - 10
        upper_thermal_threshold = self.max_ground_temp_K + 10

        photo_shoot_config.set_lower_thermal_threshold(lower_thermal_threshold)
        photo_shoot_config.set_upper_thermal_threshold(upper_thermal_threshold)

        # Set only one pose at (0,0,35)
        pose = gzm.Pose3d(0, 0, 35, 0.0, 1.57, 0.0)
        photo_shoot_config.add_pose(pose)

        self.world_config.add_plugin(photo_shoot_config)

    def configure_forest_ground_truth(self):
        self.forest_config = ForestConfig()
        forest_config = self.forest_config

        # Set forest generation but with zero trees
        forest_config.set_generate(True)
        forest_config.set_trees(0)
        forest_config.set_ground_texture(0)
        forest_config.set_direct_spawning(True)
        forest_config.set_texture_size(35)

        # Use the same ground thermal texture
        forest_config.set_ground_thermal_texture(
            os.path.join(self.thermal_texture_dir, self.thermal_texture),
            self.min_ground_temp_K,  # Minimal temperature in Kelvin
            self.max_ground_temp_K,  # Maximal temperature in Kelvin
        )

        self.world_config.add_plugin(forest_config)


if __name__ == "__main__":
    simulation_runner = SimulationRunner()
    simulation_runner.run()
