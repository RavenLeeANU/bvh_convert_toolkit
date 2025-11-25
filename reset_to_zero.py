import argparse
import os
import os
from os import listdir

def parse_bvh_file(file_path):
    """
    解析BVH文件，分离头部信息和帧数据
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 找到MOTION部分
    motion_start = None
    for i, line in enumerate(lines):
        if line.strip() == 'MOTION':
            motion_start = i
            break
    
    if motion_start is None:
        raise ValueError("BVH文件格式错误：未找到MOTION部分")
    
    # 分离头部和帧数据
    header = lines[:motion_start + 2]  # 包括MOTION和Frames行
    frame_lines = lines[motion_start + 3:]  # 帧数据
    
    # 解析帧数
    frames_line = lines[motion_start + 1].strip()
    if frames_line.startswith('Frames:'):
        num_frames = int(frames_line.split(':')[1].strip())
    else:
        raise ValueError("BVH文件格式错误：未找到Frames信息")
    
    # 解析帧时间
    frame_time_line = lines[motion_start + 2].strip()
    if frame_time_line.startswith('Frame Time:'):
        frame_time = float(frame_time_line.split(':')[1].strip())
    else:
        raise ValueError("BVH文件格式错误：未找到Frame Time信息")
    
    # 解析帧数据
    frames = []
    for line in frame_lines:
        line = line.strip()
        if line:
            frame_data = [float(x) for x in line.split()]
            frames.append(frame_data)
    
    return header, frames, num_frames, frame_time

def process_bvh_frames(frames):
    """
    处理帧数据：每一帧减去第一帧的xy偏移
    """
    if len(frames) == 0:
        return frames
    
    # 获取第一帧的根节点位置（前3个值：x, y, z）
    first_frame_root_pos = frames[0][:3]
    x_offset, y_offset, z_offset = first_frame_root_pos
    
    processed_frames = []
    
    for i, frame in enumerate(frames):
        processed_frame = frame.copy()
        
        # 只调整xz位置，保持y轴高度不变
        processed_frame[0] -= x_offset  # x坐标
        processed_frame[2] -= z_offset  # z坐标
        
        
        processed_frames.append(processed_frame)
    
    return processed_frames

def write_bvh_file(output_path, header, processed_frames, num_frames, frame_time):
    """
    将处理后的BVH数据写入文件
    """
    with open(output_path, 'w') as file:
        # 写入头部信息
        file.writelines(header)
        
        # 写入帧数据
        for frame in processed_frames:
            frame_str = ' '.join([f'{x:.6f}' for x in frame])
            file.write(frame_str + '\n')

def reset_file_batch():
    input_dir = "/home/liwenrui/liwenrui/Projects/GMR/inputs/xsens/1119/female/"
    output_dir = "/home/liwenrui/liwenrui/Projects/GMR/inputs/xsens/1119/female/reset/"

    files = sorted([f for f in listdir(input_dir) if f.endswith(".bvh") and '_XYZ' not in f])
    print(files, len(files))

    for f in files:
        
        input_path = os.path.join(input_dir,f)
        output_path = os.path.join(output_dir,f)

        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            print(f"错误：文件 '{input_path}' 不存在")
            return
        
        try:
            # 解析BVH文件
            print(f"正在解析BVH文件: {input_path}")
            header, frames, num_frames, frame_time = parse_bvh_file(input_path)
            
            print(f"找到 {num_frames} 帧数据")
            
            # 处理帧数据
            print("正在处理帧数据...")
            processed_frames = process_bvh_frames(frames)
            
            # 写入输出文件
            print(f"正在写入输出文件: {output_path}")
            write_bvh_file(output_path, header, processed_frames, num_frames, frame_time)
            
            print("处理完成！")
            print(f"第一帧位置已调整到坐标原点，所有帧的xy偏移已修正")
            
        except Exception as e:
            print(f"处理过程中发生错误: {str(e)}")

if __name__ == "__main__":
    reset_file_batch()