import argparse
import sys
import os
import utils.bvh as bvh
import bpy
import numpy as np
import os
from os import listdir


def mirror(input_path):

    mocap_files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.bvh') and 'XYZ' in f and '_mirror' not in f and '_m' not in f])
    print(mocap_files)
   
    for f in mocap_files:
        # Get the mirror file name
        mirror_f = f.split('.bvh')[0] + '_mirror.bvh'
        print(mirror_f)

        # Load an animation from a BVH file
        anim = bvh.load(filepath=f)
        # Skel offsets must be symmetric!!!

        # Mirror the animation
        anim_mirror = anim.mirror()

        # Save the animation
        bvh.save(
            filepath=mirror_f,
            anim=anim_mirror
    )


def mirror_xsens(input_path):
    XSens2XYZ(input_path)
    mirror(input_path)
    XYZ2Xsens(input_path)
    delete_mirror_files(input_path)

def XSens2XYZ(input_path):
    files = sorted([f for f in listdir(input_path) if f.endswith(".bvh") and '_XYZ' not in f])
    print(files, len(files))

    for f in files:
        source_path = os.path.join(input_path, f)
        dump_path = source_path[:-4] + '_XYZ.bvh'

        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        bpy.ops.import_anim.bvh(filepath=source_path, update_scene_fps=True,
                                # for lafan1 rotation change axis
                                # axis_forward='-Z',
                                # for motorica rotation change axis
                                # axis_up='-Y',
                                # for xsens
                                axis_up='Y'
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
                                rotate_mode='XYZ',
                                root_transform_only=True,
                                )
        bpy.data.actions.remove(bpy.data.actions[-1])

        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()


def XYZ2Xsens(input_path):
    files = sorted([f for f in listdir(input_path) if f.endswith(".bvh") and '_XYZ_mirror' in f])
    print(files, len(files))

    for f in files:
        source_path = os.path.join(input_path, f)
        dump_path = source_path.replace("_XYZ_mirror.bvh",'_m.bvh')

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

def delete_mirror_files(directory_path):
    """
    删除指定目录下所有文件名包含'mirror'的文件
    
    Args:
        directory_path (str): 要搜索的目录路径
    """
    if not os.path.exists(directory_path):
        print(f"错误: 路径 '{directory_path}' 不存在")
        return
    
    if not os.path.isdir(directory_path):
        print(f"错误: '{directory_path}' 不是一个目录")
        return
    
    deleted_files = []
    error_files = []
    
    # 遍历目录中的所有文件
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if 'mirror' in file.lower():  # 不区分大小写匹配
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    deleted_files.append(file_path)
                    print(f"已删除: {file_path}")
                except Exception as e:
                    error_files.append((file_path, str(e)))
                    print(f"删除失败: {file_path} - 错误: {e}")
    
    # 输出结果摘要
    print(f"\n操作完成:")
    print(f"成功删除 {len(deleted_files)} 个文件")
    print(f"删除失败 {len(error_files)} 个文件")
    
    if deleted_files:
        print("\n已删除的文件:")
        for file in deleted_files:
            print(f"  - {file}")
    
    if error_files:
        print("\n删除失败的文件:")
        for file, error in error_files:
            print(f"  - {file}: {error}")


def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='bvh格式转换工具')
    
    # 添加命令行参数
    parser.add_argument('--input_path', '-i', type=str, required=True,
                       help='输入文件夹路径')
    # parser.add_argument('--output_path', '-o', type=str, required=True,
    #                    help='输出文件夹路径')
    parser.add_argument('--convert_type', '-t', type=str, required=True,
                       choices=['XSens2XYZ', 'MirrorXYZ', 'MirrorXSens','XYZ2XSens'],
                       help='转换类型: XSens转GMP训练数据, 镜像GMP训练数据, 镜像XSens数据, GMP训练数据转XSens')
    
    # 解析参数
    args = parser.parse_args()
    
    # 根据转换类型调用对应的函数
    function_mapping = {
        'MirrorXYZ': mirror,
        'XSens2XYZ': XSens2XYZ,
        'XYZ2XSens': XYZ2Xsens,
        'MirrorXSens' : mirror_xsens
    }
    
    # 获取对应的处理函数
    process_func = function_mapping.get(args.convert_type)
    
    if process_func:
        try:
            # 调用处理函数
            process_func(args.input_path)
            print(f"转换完成! 输入类型: {args.input_file_type}")
        except Exception as e:
            print(f"处理过程中发生错误: {str(e)}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"不支持的转换类型: {args.convert_type}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()