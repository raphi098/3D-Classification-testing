import os
import numpy as np
import warnings
import pickle
import open3d as o3d
from tqdm import tqdm
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):

    def __init__(self, root_dir, process_data=False, split='train'):
        self.root = root_dir
        self.process_data = process_data
        self.categories = sorted([name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))])
        self.num_categories = len(self.categories)
        self.split = split
        
        if not os.path.exists(os.path.join(self.root, "shape_names.txt")):
            with open(os.path.join(self.root, "shape_names.txt"), "w") as f:
                f.write("\n".join(self.categories))
        
        self.categories_file = os.path.join(self.root, 'shape_names.txt')
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
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[self.split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], self.split, shape_ids[self.split][i])) for i in range(len(shape_ids[self.split]))]

        self.save_path = os.path.join(self.root, 'custom.dat')

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

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)
        

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

        return point_set, label[0]
    
    def __getitem__(self, index):
        return self._get_item(index)
    
    def get_filenames(self, split):
        file_names = []
        for category in self.categories:
            if os.path.isdir(os.path.join(self.root, category)):
                for file in os.listdir(os.path.join(self.root, category, split)):
                    file_names.append(file)
        return file_names
    
if __name__ == "__main__":
    root = "pc_dataset_v1"
    from types import SimpleNamespace
    args = SimpleNamespace(num_point=1024, use_uniform_sample=False, use_normals=False, num_category=40)
    test = PointCloudDataset(root)