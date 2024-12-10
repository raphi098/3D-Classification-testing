import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool

class StlToPointCloud:
    def __init__(self, path_data, output_dir, train_test_split=0.8):
        self.path_data = path_data
        self.train_test_split = train_test_split
        self.output_dir = output_dir
        self.folder_names = []

        if not os.path.exists("Data_prepared"):
            os.makedirs("Data_prepared")

        print(f"Creating folder {self.output_dir}")
        self.folder_names = [folder for folder in os.listdir(self.path_data) if os.path.isdir(os.path.join(self.path_data, folder))]
        print(self.folder_names)
        os.makedirs(self.output_dir)
        self.create_train_test_folders()
        self.process_dataset()

    def create_train_test_folders(self):
        for folder_name in self.folder_names:
            folder_name = folder_name.replace(" ", "_")
            if not os.path.exists(os.path.join(self.output_dir, folder_name)):
                os.makedirs(os.path.join(self.output_dir, folder_name))
                print(f"Created folder {folder_name}")
            if not os.path.exists(os.path.join(self.output_dir, folder_name, "train")):
                os.makedirs(os.path.join(self.output_dir, folder_name, "train"))
                print(f"Created folder {folder_name}/train")
            if not os.path.exists(os.path.join(self.output_dir, folder_name, "test")):
                os.makedirs(os.path.join(self.output_dir, folder_name, "test"))
                print(f"Created folder {folder_name}/test")

    def process_dataset(self):
        # Create a single Pool to reuse workers
        with Pool() as pool:
            for folder in tqdm(self.folder_names, desc="Processing folders"):
                folder_path = os.path.join(self.path_data, folder)

                count_files = len(os.listdir(os.path.join(self.path_data, folder)))
                split_index = int(count_files * self.train_test_split)
                train_files = os.listdir(os.path.join(self.path_data, folder))[:split_index]
                test_files = os.listdir(os.path.join(self.path_data, folder))[split_index:]

                for split in ["train", "test"]:
                    files = train_files if split == "train" else test_files
                    file_paths = [(os.path.join(folder_path, file),
                                os.path.join(self.output_dir, folder.replace(" ", "_"), split, file.replace(" ", "_")))
                                for file in files]

                    # Use the persistent Pool instance
                    list(tqdm(pool.imap(self.process_mesh_wrapper, file_paths),
                            total=len(file_paths), desc=f"Processing files in {split}", leave=False))

    @staticmethod
    def process_mesh_wrapper(args):
        file_path, destination_path_mesh = args
        StlToPointCloud.process_mesh(file_path, destination_path_mesh)

    @staticmethod
    def process_mesh(file_path, destination_path_mesh):
        mesh = o3d.io.read_triangle_mesh(file_path)
        if mesh.is_empty():
            print(f"Empty mesh: {file_path}")
            return
        # Normalize as unit ball with radius 1
        points = np.asarray(mesh.vertices)
        centroid = points.mean(axis=0)
        points -= centroid
        max_distance = np.linalg.norm(points, axis=1).max()
        points /= max_distance
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

        vertices = np.asarray(mesh.vertices)
        centroid = vertices.mean(axis=0)
        vertices_centered = vertices - centroid
        mesh.vertices = o3d.utility.Vector3dVector(vertices_centered)

        o3d.io.write_triangle_mesh(destination_path_mesh, mesh)


if __name__ == "__main__":
    path_data = os.path.join(os.getcwd(), "Data_raw", "test_dataset")
    unit_ball = True
    output_dir = os.path.join(os.getcwd(), "Data_prepared")
    stl_to_pc = StlToPointCloud(path_data, unit_ball, output_dir)
