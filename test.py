import os 
from DatasetLoader.Stl_to_Pointcloud import StlToPointCloud

def test_stl_dataset_to_pc():
    dataset_path = os.path.join(os.getcwd(), "Data_raw", "test_dataset" )
    number_of_points = 1024
    stl_to_pc = StlToPointCloud(dataset_path, number_of_points)

