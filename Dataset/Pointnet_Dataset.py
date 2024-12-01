import os
import numpy as np
import warnings
import pickle
import open3d as o3d
from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class PointnetDataset(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
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
        print(self.classes)

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
        print(self.num_category, split, self.npoints)
        if self.uniform:
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
                    point_set = o3d.io.read_point_cloud(fn[1])
                    point_set = np.asarray(point_set.points).astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

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
            point_set = o3d.io.read_point_cloud(fn[1])
            point_set = np.asarray(point_set.points).astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
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
    test = PointnetDataset(root, args, "train", process_data=True)