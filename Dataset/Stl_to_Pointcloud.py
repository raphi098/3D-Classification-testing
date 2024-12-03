import open3d as o3d
import numpy as np
import os
from tqdm import tqdm

class StlToPointCloud:
    def __init__(self, dataset_path, number_of_points, unit_ball, train_test_split=0.8):
        self.dataset_path = dataset_path
        self.number_of_points = number_of_points
        self.train_test_split = train_test_split
        self.unit_ball = unit_ball
        self.pc_dataset_path = os.path.join("Data_prepared", f"{os.path.basename(dataset_path)}_{number_of_points}_points_unitball_{unit_ball}")
        self.folder_names = []

        if not os.path.exists("Data_prepared"):
            os.makedirs("Data_prepared")
        if not os.path.exists(self.pc_dataset_path):
            print(f"Creating folder {self.pc_dataset_path}")
            self.folder_names = [folder for folder in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, folder))]
            print(self.folder_names)
            os.makedirs(self.pc_dataset_path)
            self.create_train_test_folders() 
            self.convert_dataset_to_pc()
        else:
            print(f"Folder {self.pc_dataset_path} already exists")
    
    def create_train_test_folders(self):
        for folder_name in self.folder_names:
            #Replace spaces in folder names with underscores
            folder_name = folder_name.replace(" ", "_")
            if not os.path.exists(os.path.join(self.pc_dataset_path, folder_name)):
                os.makedirs(os.path.join(self.pc_dataset_path, folder_name))
                print(f"Created folder {folder_name}")
            if not os.path.exists(os.path.join(self.pc_dataset_path, folder_name, "train")):
                os.makedirs(os.path.join(self.pc_dataset_path, folder_name, "train"))
                print(f"Created folder {folder_name}/train")
            if not os.path.exists(os.path.join(self.pc_dataset_path, folder_name, "test")):
                os.makedirs(os.path.join(self.pc_dataset_path, folder_name, "test"))
                print(f"Created folder {folder_name}/test")

    def convert_dataset_to_pc(self):
        for folder in tqdm(self.folder_names, desc="Processing folders"):
            folder_path = os.path.join(self.dataset_path, folder)

            # Count files for splitting the dataset
            count_files = len(os.listdir(os.path.join(self.dataset_path, folder)))
            split_index = int(count_files * self.train_test_split)
            train_files = os.listdir(os.path.join(self.dataset_path, folder))[:split_index]
            test_files = os.listdir(os.path.join(self.dataset_path, folder))[split_index:]

            for split in ["train", "test"]:
                if split == "train":
                    files = train_files
                else:
                    files = test_files

                for file in tqdm(files, desc=f"Processing files in {split}", leave=False):
                    file_path = os.path.join(folder_path, file)
                    destination_path = os.path.join(self.pc_dataset_path, folder.replace(" ", "_"), split, file.replace(".stl", ".pcd").replace(" ", "_"))
                    self.convert_to_pc(file_path, destination_path)
    
    def convert_to_pc(self, file_path, destination_path):
        mesh = o3d.io.read_triangle_mesh(file_path)
        if self.unit_ball:
            # Normalize mesh to sphere with radius = 1
            # Step 1: Translate to zero mean
            points = np.asarray(mesh.vertices)
            centroid = points.mean(axis=0)
            points -= centroid

            # Step 2: Scale to fit within a unit sphere
            max_distance = np.linalg.norm(points, axis=1).max()
            points /= max_distance

            # Update mesh with normalized points
            mesh.vertices = o3d.utility.Vector3dVector(points)

            # Step 3: Compute normals (required for STL format)
            mesh.compute_triangle_normals()  # Compute face normals
            mesh.compute_vertex_normals()  # Optional: compute vertex normals

            # Center the mesh
            vertices = np.asarray(mesh.vertices)
            centroid = vertices.mean(axis=0)
            vertices_centered = vertices - centroid
            mesh.vertices = o3d.utility.Vector3dVector(vertices_centered)

        pc = mesh.sample_points_poisson_disk(number_of_points=self.number_of_points)
        o3d.io.write_point_cloud(destination_path, pc)


if __name__ == "__main__":
    dataset_path = os.path.join(os.getcwd(), "Data_raw", "test_dataset" )
    number_of_points = 1024
    stl_to_pc = StlToPointCloud(dataset_path, number_of_points)