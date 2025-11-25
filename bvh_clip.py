import os
import utils.bvh as bvh


data_dir = '../mirror'
mocap_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.bvh') and 'XYZ' not in f and 'mirror' not in f])
print(mocap_files)
# mocap_files = ["../data/xsens-mocap/walk8_fast.bvh"]

def clip_frame(source, destination, start=0, end=-1):
    print(f'Processing {source} to {destination}')
    anim = bvh.load(source)
    
for f in mocap_files:
    # Get the mirror file name
    mirror_f = f.split('.bvh')[0] + '_clip.bvh'
    clip_frame(f, mirror_f, start=0, end=150)

