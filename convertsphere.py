import open3d as o3d
import os
import numpy as np
import copy

# Define the input path
input_path = "all_100_files"

# Create the output directory for normalized meshes
output_path = "sphere_all_100_files"
os.makedirs(output_path, exist_ok=True)

# List all folders in the input directory
folders = os.listdir(input_path)

for folder in folders:
    folder_path = os.path.join(input_path, folder)
    output_folder_path = os.path.join(output_path, folder)
    os.makedirs(output_folder_path, exist_ok=True)  # Create subfolder for normalized files

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        output_file_path = os.path.join(output_folder_path, file)

        # Read the mesh
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_vertices():
            print(f"File {file_path} has no vertices. Skipping.")
            continue

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

        # Save the normalized mesh
        o3d.io.write_triangle_mesh(output_file_path, mesh)
        print(f"Normalized and saved: {output_file_path}")

print("Normalization completed.")
