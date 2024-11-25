import os
import open3d as o3d
import numpy as np
import multiprocessing
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class MultiviewDataset:
    def __init__(self, data_path, num_views=12, train_test_split=0.8):
        self.data_path = data_path
        self.output_dir = os.path.join("Data_prepared", f"{os.path.basename(data_path)}_{num_views}_views")
        self.num_views = num_views
        self.train_test_split = train_test_split
        self.classes = [c for c in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, c))]

        # Generate train and val file splits
        self.train_files, self.val_files = self.split_files()

        # Generate multiview images
        self.convert_to_multiview_parallel()

    def split_files(self):
        """Split STL files into training and validation sets."""
        train_files = []
        val_files = []

        for c in self.classes:
            category_path = os.path.join(self.data_path, c)
            if not os.path.exists(category_path):
                raise FileNotFoundError(f"Category path does not exist: {category_path}")

            # Get all STL files in the category folder
            files = [os.path.join(category_path, f) for f in os.listdir(category_path) if f.endswith(".stl")]
            if not files:
                print(f"No STL files found in category path: {category_path}")
                continue

            # Split into train and validation sets
            train, val = train_test_split(files, test_size=1 - self.train_test_split, random_state=42)
            train_files.extend([(c, f) for f in train])
            val_files.extend([(c, f) for f in val])

        return train_files, val_files

    @staticmethod
    def generate_views(mesh_path, output_folder, viewpoints, visible=False):
        """Generate multiview images for a given STL file."""
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if not mesh.has_triangle_normals():
            mesh.compute_vertex_normals()

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=visible)
        vis.add_geometry(mesh)

        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)

        for i, front in enumerate(viewpoints):
            ctr.set_front(front.tolist())
            ctr.set_lookat(mesh.get_center())
            ctr.set_up([0, 0, 1])
            vis.poll_events()
            vis.update_renderer()

            output_image_path = os.path.join(output_folder, f"{i}.png")
            vis.capture_screen_image(output_image_path)

        vis.destroy_window()

    def convert_to_multiview_parallel(self):
        """Convert STL files to multiview images in parallel."""
        tasks = []
        viewpoints = self.get_viewpoints()

        for phase, files in [("train", self.train_files), ("val", self.val_files)]:
            for class_name, mesh_path in files:
                output_folder = os.path.join(self.output_dir, phase, class_name, os.path.splitext(os.path.basename(mesh_path))[0])
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                tasks.append((mesh_path, output_folder, viewpoints))

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            list(tqdm(pool.imap_unordered(self._process_view_task, tasks), total=len(tasks), desc="Generating multiviews"))

    @staticmethod
    def _process_view_task(args):
        """Process a single view generation task."""
        mesh_path, output_folder, viewpoints = args
        MultiviewDataset.generate_views(mesh_path, output_folder, viewpoints)

    def get_viewpoints(self):
        """Get camera viewpoints based on the number of views."""
        if self.num_views == 12:
            return [np.array([np.cos(np.radians(i * 360 / self.num_views)), np.sin(np.radians(i * 360 / self.num_views)), 0]) for i in range(self.num_views)]
        elif self.num_views == 20:
            phi = (1 + np.sqrt(5)) / 2
            vertices = [
                [1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1],
                [1, 1, -1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1],
                [0, 1 / phi, phi], [0, -1 / phi, phi], [0, 1 / phi, -phi], [0, -1 / phi, -phi],
                [1 / phi, phi, 0], [-1 / phi, phi, 0], [1 / phi, -phi, 0], [-1 / phi, -phi, 0],
                [phi, 0, 1 / phi], [-phi, 0, 1 / phi], [phi, 0, -1 / phi], [-phi, 0, -1 / phi]
            ]
            vertices = np.array(vertices)
            vertices /= np.linalg.norm(vertices[0])
            return vertices
        else:
            raise ValueError(f"Unsupported number of views: {self.num_views}")


if __name__ == "__main__":
    dataset = MultiviewDataset(r"Data_raw\test_dataset", num_views=12, train_test_split=0.8)
