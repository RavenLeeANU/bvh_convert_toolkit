import bpy
import numpy as np
import os
from os import listdir

# data_path = '../mirror'
# data_path = '/home/liwenrui/Data/motorica_bvh/'
data_path = '/home/liwenrui/liwenrui/Projects/retarget/data/1122/'

files = sorted([f for f in listdir(data_path) if f.endswith(".bvh")])
print(files, len(files))

for f in files:
    source_path = os.path.join(data_path, f)
    if "_mirror_XYZ" in source_path:
        dump_path = source_path.replace("_mirror_XYZ.bvh",'_mirror.bvh')
    else:
        dump_path = source_path.replace("_XYZ.bvh",'.bvh')

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    bpy.ops.import_anim.bvh(filepath=source_path, update_scene_fps=True,
                            rotate_mode='XYZ',
                            # for lafan1 rotation change axis
                            axis_forward='Z',
                            # for motorica rotation change axis
                            # axis_up='-Y',
                            # for xsens
                            axis_up='-Y'
                            )

    frame_start = 9999
    frame_end = -9999
    action = bpy.data.actions[-1]
    if action.frame_range[1] > frame_end:
        frame_end = action.frame_range[1]
    if action.frame_range[0] < frame_start:
        frame_start = action.frame_range[0]

    frame_end = np.max([1, frame_end])
    bpy.ops.export_anim.bvh(filepath=dump_path,
                            frame_start=int(frame_start),
                            frame_end=int(frame_end),
                            rotate_mode='YXZ',
                            root_transform_only=True,
                            )
    bpy.data.actions.remove(bpy.data.actions[-1])

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    print(source_path + " processed.")
