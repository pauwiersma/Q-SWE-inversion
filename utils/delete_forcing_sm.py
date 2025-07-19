import os
import glob
import shutil

files = glob.glob("/home/pwiersma/scratch/Data/ewatercycle/experiments/data/input/*")
print(len(files))
for i,f in enumerate(files):
    if len(os.path.basename(f))>60:
        os.remove(f)
    elif 'tt_scale' in os.path.basename(f):
        os.remove(f)
    elif '_ms' in os.path.basename(f):
        os.remove(f)
    if i%1000==0:
        print(i)