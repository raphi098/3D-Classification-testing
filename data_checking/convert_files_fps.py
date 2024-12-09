import torch
import open3d as o3d
import numpy as np
import os

def farthest_point_sample(point, npoint):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to PyTorch tensor and move to device
    point = torch.tensor(point, device=device, dtype=torch.float32)
    N, D = point.shape
    xyz = point[:, :3] 

    sampled_indices = torch.zeros(npoint, dtype=torch.long, device=device)
    distance = torch.ones(N, device=device) * 1e10 
    farthest = torch.randint(0, N, (1,), device=device)

    for i in range(npoint):
        sampled_indices[i] = farthest
        centroid = xyz[farthest, :].unsqueeze(0)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist) 

        farthest = torch.argmax(distance, dim=-1)

    sampled_points = point[sampled_indices] 
    return sampled_points.cpu().numpy(), sampled_indices.cpu().numpy()

# Paths for input and output
input_path = "failed_files"
output_path = "failed_files_fps"
os.makedirs(output_path, exist_ok=True)

# Process each file
file_names = os.listdir(input_path)
for f in file_names:
    input_file_path = os.path.join(input_path, f)
    output_file_path = os.path.join(output_path, f.replace('.stl', '_fps.ply'))
    
    # Read the mesh
    mesh = o3d.io.read_triangle_mesh(input_file_path)
    if not mesh.has_vertices():
        print(f"Mesh in {f} has no vertices, skipping.")
        continue
    
    # Get the vertices as point cloud
    point_set = np.asarray(mesh.vertices).astype(np.float32)
    
    # Perform farthest point sampling
    point_set, sampled_indices = farthest_point_sample(point_set, 1024)
    
    # Convert to Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set)
    
    # Save the sampled point cloud
    o3d.io.write_point_cloud(output_file_path, pcd)
    print(f"Saved sampled point cloud to {output_file_path}")
