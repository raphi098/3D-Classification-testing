import os
import numpy as np
import warnings
import pickle
import open3d as o3d
from tqdm import tqdm
from torch.utils.data import Dataset
import multiprocessing as mp
import torch

warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

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

class PointnetDataset(Dataset):
    def __init__(self, num_points, root, args, split, process_data=False):
        self.root = root
        self.npoints = num_points
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

        if not os.path.exists(os.path.join(self.root, "shape_names.txt")):
            shapes = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
            with open(os.path.join(self.root, "shape_names.txt"), "w") as f:
                f.write("\n".join(shapes))

        self.categories_file = os.path.join(self.root, 'shape_names.txt')

        self.categories = [line.rstrip() for line in open(self.categories_file)]
        self.classes = dict(zip(self.categories, range(len(self.categories))))

        if not os.path.exists(os.path.join(self.root, "train.txt")):
            file_names = self.get_filenames("train")
            with open(os.path.join(self.root, "train.txt"), "w") as f:
                f.write("\n".join(file_names))
        
        if not os.path.exists(os.path.join(self.root, "test.txt")):
            file_names = self.get_filenames("test")
            with open(os.path.join(self.root, "test.txt"), "w") as f:
                f.write("\n".join(file_names))

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'test.txt'))]

        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], split, shape_ids[split][i])) for i in range(len(shape_ids[split]))]

        if self.use_normals:
            self.save_path = os.path.join(self.root, 'custom%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(self.root, 'custom%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))
        print('The size of %s data is %d' % (split, len(self.datapath)))
        if self.process_data:
            if not os.path.exists(self.save_path):
                print("Processing data ...")
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath))):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    mesh = o3d.io.read_triangle_mesh(fn[1])

                    point_set = np.asarray(mesh.vertices).astype(np.float32)
                    point_set, sampled_indices = farthest_point_sample(point_set, self.npoints)

                    if self.use_normals:
                        if not mesh.has_vertex_normals():
                            mesh.compute_vertex_normals()
                        vertex_normals = np.asarray(mesh.vertex_normals).astype(np.float32)

                        # Sample the normals corresponding to the sampled points
                        vertex_normals = vertex_normals[sampled_indices]

                        # Concatenate the points and normals
                        point_set = np.concatenate((point_set, vertex_normals), axis=1)


                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def get_filenames(self, split):
        file_names = []
        for category in self.categories:
            if os.path.isdir(os.path.join(self.root, category)):
                for file in os.listdir(os.path.join(self.root, category, split)):
                    file_names.append(file)
        return file_names


    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            mesh = o3d.io.read_triangle_mesh(fn[1])

            if self.uniform:
                point_set = np.asarray(mesh.vertices).astype(np.float32)
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = mesh.sample_points_poisson_disk(self.npoints)
                point_set = np.asarray(point_set.points).astype(np.float32)
                point_set = point_set[0:self.npoints, :]
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)
    
if __name__ == "__main__":
    root = os.path.join("data", "pc_dataset_v1")
    from types import SimpleNamespace
    args = SimpleNamespace(num_point=1024, use_uniform_sample=False, use_normals=False, num_category=40)
    test = PointnetDataset(1024, root, args, "train", process_data=True)
