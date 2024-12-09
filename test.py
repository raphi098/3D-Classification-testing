import os
import shutil
# Path to the point cloud file
files = ['1gliedrig_flach_(10038).stl', 
        '1gliedrig_flach_(10039).stl', 
        '1gliedrig_glatt_(15680).stl', 
        '1gliedrig_glatt_(15681).stl', 
        '1gliedrig_glatt_(6943).stl', 
        '1gliedrig_glatt_(6944).stl', 
        '1gliedrig_kappe_front_(136).stl', 
        '1gliedrig_sattel_(1038).stl', 
        '1gliedrig_sattel_(1040).stl', 
        '1gliedrig_sattel_(2551).stl', 
        '1gliedrig_stufe_(1037).stl', 
        '1gliedrig_stufe_(1038).stl', 
        '1gliedrig_stufe_(1041).stl', 
        '1gliedrig_stufe_(1042).stl', 
        '1gliedrig_kappe_front_(135).stl', 
        '1gliedrig_rundung_(1037).stl', 
        '1gliedrig_rundung_(1039).stl', 
        '1gliedrig_sattel_(2527).stl']

path = os.path.join("Data_raw", "1gliedrig_50_files")

os.makedirs("failed_files", exist_ok=True)

for file in files:
    folder = file.split("_")[0] + " " + file.split("_")[1]
    file = file.replace("_"," ", 1)
    path_file = os.path.join(path, folder, file)
    
    try:
        shutil.copy(path_file, os.path.join("failed_files", file))
    except FileNotFoundError:
        print(f"File {path_file} not found")

# moves files back to the original folder
# for file in files:
#     folder = file.split("_")[0] + " " + file.split("_")[1]
#     file = file.replace("_"," ", 1)
#     path_file = os.path.join("failed_files", file)
    
#     shutil.move(path_file, os.path.join(path, folder, file))
    