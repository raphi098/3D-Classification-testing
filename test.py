import open3d as o3d
import os
import numpy as np
# Path to the point cloud file
path = os.path.join("Data_prepared", "1gliedrig_100_files_1024_points", "1gliedrig", "test", "1gliedrig_(1007).pcd")

# Read the point cloud
point = o3d.io.read_point_cloud(path)

# Estimate the normals
point.estimate_normals()

# Access the normals
normals = point.normals

# Print the normals
print(np.asarray(normals))
