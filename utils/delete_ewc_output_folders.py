import os
import glob
import shutil
folders = os.listdir("/home/pwiersma/scratch/Data/ewatercycle/experiments/ewc_outputs")
print(len(folders))
for i,f in enumerate(folders):
    try:
        shutil.rmtree(os.path.join(
            "/home/pwiersma/scratch/Data/ewatercycle/experiments/ewc_outputs",f))
    except:
        print(f+"couldn't be removed")
        continue
    if i%1000==0:
        print(i)