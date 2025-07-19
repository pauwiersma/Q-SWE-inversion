
import os
import glob
import shutil
import zipfile
import datetime
# ('clusters_m' in f) | ('test' in f) &

def zip_folder(zipf, folder_path, base_folder):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, base_folder)
            zipf.write(file_path, arcname)
def get_creation_date(path):
    # Get the creation time
    creation_time = os.path.getctime(path)
    # Convert to a human-readable format
    creation_date = datetime.datetime.fromtimestamp(creation_time)
    return creation_date

folders = os.listdir("/home/pwiersma/scratch/Data/ewatercycle/experiments/data/")
zip_file = "/home/pwiersma/scratch/Data/ewatercycle/experiments/output_folders_13_9.zip"

print(f"Number of folders: {len(folders)}")

with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for i, folder in enumerate(folders):
        if (not 'joint' in folder) and (not 'input' in folder):
            creation_date = get_creation_date(os.path.join(
                "/home/pwiersma/scratch/Data/ewatercycle/experiments/data/", folder))
            if creation_date<datetime.datetime(2024,4,1):
                folder_path = os.path.join("/home/pwiersma/scratch/Data/ewatercycle/experiments/data/", folder)
                zip_folder(zipf, folder_path, "/home/pwiersma/scratch/Data/ewatercycle/experiments/data/")
                shutil.rmtree(folder_path)
                # print(folder_path)
        if i % 100 == 0:
            print(f"Processed {i} folders")


            
        # if i ==10:
        #     break
# for i, folder in enumerate(folders):
#     if (not 'joint' in folder) and (not 'input' in folder):
#         creation_date = get_creation_date(os.path.join(
#             "/home/pwiersma/scratch/Data/ewatercycle/experiments/data/", folder))
#         if creation_date<datetime.datetime(2024,4,1):
#             folder_path = os.path.join("/home/pwiersma/scratch/Data/ewatercycle/experiments/data/", folder)
#             shutil.rmtree(folder_path)
#             print(folder_path)
#     if i % 100 == 0:
#         print(f"Processed {i} folders")

# folders = os.listdir("/home/pwiersma/scratch/Data/ewatercycle/experiments/data/")
# zip_file = "/home/pwiersma/scratch/Data/ewatercycle/experiments/output_folders_12_9.zip"
# print(len(folders))
# with zipfile.ZipFile(zip_file, 'w') as zipf:
#     for i,f in enumerate(folders):
#         if (not 'joint' in f) & (not 'input' in f):
#             zipf.write(os.path.join(
#                 "/home/pwiersma/scratch/Data/ewatercycle/experiments/data/",f), f)
#         if i%100==0:
#             print(i)


# for i,f in enumerate(folders):
#     if (not 'joint' in f) & (not 'input' in f):
#         try:
#             shutil.rmtree(os.path.join(
#                 "/home/pwiersma/scratch/Data/ewatercycle/experiments/data/",f))
#         except:
#             print(f+"couldn't be removed")
#             continue
#     else:
#         print(f)
#     if i%100==0:
#         print(i)