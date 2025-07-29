# v3 1.7 用numba+njit cpu加速， 模型在NPU上运行。


# import cProfile
# import pstats
# import snakeviz
 

# profiler = cProfile.Profile()
# profiler.enable()


import torch_npu
from pathlib import Path

import os, re, sys, cv2
import numpy as np
from astropy.io import fits
from numba import cuda,njit, prange

# sys.path.append('/mnt/volume/userdata/zjt/Drafts/DRAFTS/')
sys.path.append('/mnt/volume/userdata/zjt/dev/Projects/DRAFTS')
sys.path.append('/mnt/volume/userdata/zjt/dev/Projects/DRAFTS/Drafts-npu')
from utils.numpy_save_load import save_data, load_data


import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('default')
sns.set_color_codes()


import time
import torch
device = torch.device('npu' if torch_npu.npu.is_available() else 'cpu')

# 简单分析查看了，这两个模块里都不涉及到cuda jit编译，只涉及到pytorch的模块。
from centernet_utils import get_res
from centernet_model import centernet

from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops
import copy
import inspect
import os

# 保存处理后的数据
def save_processed_data(slice_data, save_path, filename, j, time_reso, down_time_rate, down_freq_rate, DM_range, block_size, verbose=True):
    """
    保存处理后的数据到指定路径，与data_pipeline.py保持完全一致
    
    参数:
    - slice_data: 要保存的数据切片
    - save_path: 保存路径
    - filename: 基本文件名
    - j: 块索引
    - time_reso, down_time_rate, down_freq_rate, DM_range, block_size: 元数据参数
    - verbose: 是否打印详细信息
    
    返回:
    - 保存文件的完整路径
    """
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            pass
            
    output_path = os.path.join(save_path, f"{filename}_block{j}.npz")
    np.savez_compressed(
        output_path,
        data=slice_data,
        metadata=np.array([time_reso, down_time_rate, down_freq_rate, DM_range, block_size])
    )
    
    if verbose:
        print(f"保存处理块到: {output_path}")
        
    return output_path


### 读取fits文件，只保留两维数据
def load_fits_file(file_name, reverse_flag=False):

    try:
        import fitsio
        data, h  = fitsio.read(file_name, header=True)
    except:
        with fits.open(file_name) as f:
            h    = f[1].header
            data = f[1].data
    data         = data['DATA'].reshape(h['NAXIS2']*h['NSBLK'], h['NPOL'], h['NCHAN'])[:, :2, :]
    if reverse_flag: data = np.array(data[:, :, ::-1])

    return data


### 读取fits头文件，获取观测参数，并指定为全局变量
def get_obparams(file_name):

    global freq, freq_reso, time_reso, file_leng, down_freq_rate, down_time_rate
    with fits.open(file_name) as f:
        time_reso  = f[1].header['TBIN']
        freq_reso  = f[1].header['NCHAN']
        file_leng  = f[1].header['NAXIS2'] * f[1].header['NSBLK']
        freq       = f[1].data['DAT_FREQ'][0, :].astype(np.float64)
    down_freq_rate = int(freq_reso / 512)
    down_time_rate = int((49.152 * 4 / 1e6) / time_reso)


# ### 显卡ddm
# @cuda.jit
# def de_disp(dm_time, data, freq, index):
#     x, y                 = cuda.grid(2)
#     if x < dm_time.shape[1] and y < dm_time.shape[2]:
#         td_i, DM         = 0, x
#         for i in index:
#             td_i        += data[int(4.15 * DM * (freq[i]**-2 - freq[-1]**-2) * 1e3 / time_reso / down_time_rate + y), i]
#             if i == 256: dm_time[1, x, y] = td_i
#         dm_time[2, x, y] = td_i - dm_time[1, x, y]
#         dm_time[0, x, y] = td_i


# def d_dm_time_g(data, height, width):

#     freq_gpu      = cuda.to_device(np.mean(freq.reshape(freq_reso // down_freq_rate, down_freq_rate), axis=1))
#     index_gpu     = cuda.to_device(np.append(
#         np.arange(int(10  / 4096 * freq_reso // down_freq_rate), int( 650 / 4096 * freq_reso // down_freq_rate), 1),
#         np.arange(int(820 / 4096 * freq_reso // down_freq_rate), int(4050 / 4096 * freq_reso // down_freq_rate), 1)
#     )) # cuda.to_device(np.arange(0, int(freq_reso // down_freq_rate), 1))
#     dm_time_gpu, data_gpu = cuda.to_device(np.zeros((3, height, width)).astype(np.float32)), cuda.to_device(data)

#     nthreads = (8, 128)
#     nblocks  = (height // nthreads[0] + 1, width // nthreads[1] + 1)
#     de_disp[nblocks, nthreads](dm_time_gpu, data_gpu, freq_gpu, index_gpu)
#     dm_time  = dm_time_gpu.copy_to_host()

#     return dm_time

def de_disp_pro(freq, xy):

    """确保张量在 NPU 上，并调用自定义算子"""
    # print("*********************************start************************************")
    freq_tensor = torch.from_numpy(freq).npu().float()  # 移动到 NPU
    xy_tensor = torch.from_numpy(xy).npu().float()
    output_tensor = custom_ops.de_disp(freq_tensor, xy_tensor)  # 在 NPU 上执行
    # print("*********************************end************************************")

    return output_tensor.cpu().numpy()  # 如果需要 NumPy 结果
 
# # @njit(parallel=True)
# def _de_disp_numba(dm_time, data, output, index, x, y):
#     """Numba 部分：纯数值计算（假设 output 是 1D 数组）"""
#     # for x in prange(dm_time.shape[1]):
#     #     for y in range(dm_time.shape[2]):

#     td_i = 0            
#     for i in index:
#         idx = int(output[i])  # 根据实际形状调整索引
#         if 0 <= idx < data.shape[0]:
#             td_i += data[idx, i]
#         if i == 256:
#             dm_time[1, x, y] = td_i
#     dm_time[2, x, y] = td_i - dm_time[1, x, y]
#     dm_time[0, x, y] = td_i

# @njit(parallel=True)
def de_disp(dm_time, data, freq, index):
    """主函数：协调 NPU 和 Numba"""
    for x in range(1000):
        for y in range(1000):
            xy = np.zeros(2)
            xy[0] = x * 4150
            xy[1] = y
            output = de_disp_pro(freq, xy)  # 在 NPU 上计算
            output = output.astype(np.int64)
            # _de_disp_numba(dm_time, data, output, index, x, y)  # 在 CPU/Numba 上计算
            td_i = 0            
            for i in index:
                idx = int(output[i])  # 根据实际形状调整索引
                if 0 <= idx < data.shape[0]:
                    td_i += data[idx, i]
                if i == 256:
                    dm_time[1, x, y] = td_i
            dm_time[2, x, y] = td_i - dm_time[1, x, y]
            dm_time[0, x, y] = td_i


def d_dm_time_g(data, height, width):
    freq_mean = np.mean(freq.reshape(freq_reso // down_freq_rate, down_freq_rate), axis=1)
    index = np.append(
        np.arange(int(10  / 4096 * freq_reso // down_freq_rate), int(650 / 4096 * freq_reso // down_freq_rate), 1),
        np.arange(int(820 / 4096 * freq_reso // down_freq_rate), int(4050 / 4096 * freq_reso // down_freq_rate), 1)
    )
    dm_time = np.zeros((3, height, width)).astype(np.float32)

    de_disp(dm_time, data, freq_mean, index)

    return dm_time


def preprocess_img(img):

    img  = (img - np.min(img)) / (np.max(img) - np.min(img))
    img  = (img - np.mean(img)) / np.std(img)
    img  = cv2.resize(img, (512, 512))

    img  = np.clip(img, *np.percentile(img, (0.1, 99.9)))
    img  = (img - np.min(img)) / (np.max(img) - np.min(img))
    img  = plt.get_cmap('mako')(img)
    img  = img[..., :3]

    img -= [0.485, 0.456, 0.406]
    img /= [0.229, 0.224, 0.225]

    return img


def postprocess_img(img):

    img  = np.array(img).transpose(1, 2, 0)
    img *= [0.229, 0.224, 0.225]
    img += [0.485, 0.456, 0.406]
    img  = (img * 255).astype(np.uint8)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

# 计算各输出头误差
def calc_error(fp32, fp16):
    
    # 输入可以是 Tensor 或 NumPy
    fp32 = fp32.cpu().numpy() if isinstance(fp32, torch.Tensor) else fp32
    fp16 = fp16.cpu().numpy() if isinstance(fp16, torch.Tensor) else fp16

    # 使用 NumPy 计算
    mse = np.mean((fp32 - fp16.astype(np.float32)) ** 2)  # 正确转换类型
    max_diff = np.max(np.abs(fp32 - fp16.astype(np.float32)))
    return {"MSE": mse, "MaxDiff": max_diff}


 
def find_heat_peaks(
    heat: torch.Tensor,
    topk: int = 1,
    return_values: bool = False
):
    """
    找到热图中的最大值坐标（支持批量、多通道）
    
    参数:
        heat (Tensor): 输入热图，形状可为 [B, C, H, W] 或 [H, W]
        topk (int): 返回前k个最大值坐标，默认1。若为-1则返回所有可能的最大值
        return_values (bool): 是否同时返回最大值，默认False
    
    返回:
        coordinates (List[Tuple]): 坐标列表，每个元素为 (行, 列)
        values (Tensor, 可选): 对应的值，仅当return_values=True时返回
    """
    # 统一处理输入维度
    if heat.dim() == 2:
        heat = heat.unsqueeze(0).unsqueeze(0)  # [H,W] → [1,1,H,W]
    elif heat.dim() == 3:
        heat = heat.unsqueeze(1)  # [B,H,W] → [B,1,H,W]
    
    B, C, H, W = heat.shape
    coordinates = []
    values = [] if return_values else None
    
    for b in range(B):
        for c in range(C):
            current_heat = heat[b, c]  # 当前处理的单通道热图 [H,W]
            
            # 找到所有可能的最大值位置
            max_val = current_heat.max()
            mask = (current_heat == max_val)
            yx_indices = mask.nonzero()  # 所有满足条件的坐标 [N,2]
            
            # 处理topk参数
            if topk == -1:
                selected = yx_indices
            else:
                selected = yx_indices[:topk]  # 默认取第一个出现的
            
            # 转换为坐标元组并存储
            for idx in selected:
                y, x = idx.tolist()
                coordinates.append((b, c, y, x))
                if return_values:
                    values.append(max_val)
    
    if return_values:
        return coordinates, torch.tensor(values, device=heat.device)
    else:
        return coordinates
 
def plot_heatmap(
    heat: torch.Tensor,
    save_path: str = "heatmap.png",
    mark_peak: bool = True,
    dpi: int = 300,
    cmap: str = "hot",
    title: str = "Heatmap"
):
    """
    绘制热图并保存为图片
    
    参数:
        heat (Tensor): 输入热图，支持形状 [B,C,H,W]、[C,H,W]、[H,W]
        save_path (str): 图片保存路径，默认当前目录的heatmap.png
        mark_peak (bool): 是否标记最大值位置，默认True
        dpi (int): 图片分辨率，默认300
        cmap (str): 颜色映射，默认'hot'
        title (str): 图片标题，默认'Heatmap'
    """
    # 统一处理输入维度
    if heat.dim() == 4:
        heat = heat.squeeze(0).squeeze(0)  # 取第一个批次和通道 [H,W]
    elif heat.dim() == 3:
        heat = heat.squeeze(0)  # 取第一个通道 [H,W]
    elif heat.dim() == 2:
        pass  # 已经是二维
    
    # 转换为CPU numpy数组 
    heat_np = heat.cpu().detach().numpy()  
    
    import matplotlib.pyplot as plt
    # 创建画布
    plt.figure(figsize=(10, 8))
    plt.imshow(heat_np, cmap=cmap)
    plt.colorbar(label="Intensity")
    plt.title(title)
    plt.xlabel("Width")
    plt.ylabel("Height")
    
    mark_all_peaks = False
    # 标记最大值位置
    if mark_peak:
        if mark_all_peaks:  # 新增：标记所有相同最大值的位置
                    # 使用之前定义的 find_heat_peaks 函数
                    peaks = find_heat_peaks(torch.tensor(heat_np).unsqueeze(0).unsqueeze(0), topk=-1)
                    for p in peaks:
                        b, c, y, x = p
                        plt.scatter(
                            x=x, y=y, 
                            s=100, edgecolors="cyan", 
                            facecolors="none", linewidths=2,
                            zorder=2  # 确保标记在最上层
                )
        else:  # 原始逻辑：仅标记第一个最大值
            max_idx = np.unravel_index(heat_np.argmax(), heat_np.shape)
            plt.scatter(
                x=max_idx[1], y=max_idx[0], 
                s=100, edgecolors="cyan", 
                facecolors="none", linewidths=2,
                label="Peak"
            )
            plt.legend()
    
    # 保存图片
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to: {save_path}")

def analyze_diff(pred, gt, name, top_k=3):
    """差异分析函数"""
    diff = pred - gt
    abs_diff = np.abs(diff)
    
    # 获取前K个最大差异
    flat_indices = np.argpartition(abs_diff.flatten(), -top_k)[-top_k:]
    max_locations = np.unravel_index(flat_indices, diff.shape)
    max_values = diff[max_locations]
    
    # 获取前K个最小差异
    min_flat_indices = np.argpartition(abs_diff.flatten(), top_k)[:top_k]
    min_locations = np.unravel_index(min_flat_indices, diff.shape)
    min_values = diff[min_locations]

    print(f"\n{name}差异分析:")
    print(f"最大差异TOP3: 值={max_values} 位置={max_locations}")
    print(f"最小差异TOP3: 值={min_values} 位置={min_locations}")


# 从文件加载处理后的数据
def load_processed_data(file_path):
    """加载保存的处理数据和元数据"""
    data = np.load(file_path, allow_pickle=True)
    
    # 检查是否是npz文件（包含元数据）
    if file_path.endswith('.npz'):
        return data['data'], data['metadata']
    else:
        return data, None


if __name__ == '__main__':


    # Debug flags

    # 不替换 hm, wh, offset
    Debug = True

    change_load = False 
    if change_load: 
        print('替换hm, wh, offset.')
    else:
        print('未替换hm, wh, offset.')

    loaded_data_flag = False
    if loaded_data_flag: 
        print('使用已加载的数据进行推理。')
    else:
        print(f'不加载Nvidia的npy数据用于推理。')
    
    use_loaded_image = False
    if use_loaded_image: 
        print('使用已加载的img数据进行推理。')
    else:
        print(f'不使用已加载的img数据进行推理。')

    compare_error = False
    if compare_error: 
        print('逐个比较推理结果heat wh offset的误差。')
    else:
        print(f'不逐个比较推理结果的误差。')

    # 记录开始时间
    all_start_time = time.time()
    print('$主程序开始运行:$$$$')
    DM_range      = 2048
    block_size    = 8192
    det_prob      = 0.3

    base_path = '/home/zjt/DRAFTS/'

    ## 载入模型
    base_model    = 'resnet18'
    model         = centernet(model_name=base_model).to(f'npu:{device}' if isinstance(device, int) else device)
    model.load_state_dict(torch.load(base_path + 'Drafts-npu/cent_resnet18.pth', map_location=device, weights_only=True))
    model.eval()
    
    # 检查模型参数所在的设备
    device_check = next(model.parameters()).device
    print('model is on device: ', device_check)

    data_path     = base_path + 'Drafts-npu/'
    save_path     = base_path + 'Drafts-npu/results_npu_single' # results_npu_single_load_saved'
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            pass

    file_list     = np.sort([i for i in os.listdir(data_path) if i.endswith('fits')])
    # 避免添加重复fits文件
    add_fits = False
    if add_fits:
        file_list     = np.append(file_list, file_list[-1])
        print('添加重复fits文件来用于测试')
    else:
        print('不添加重复fits文件来用于测试。')

    get_obparams(data_path + file_list[0])

    ### combine file number
    dds           = (4.15 * DM_range * (freq**-2 - freq.max()**-2) * 1e3 / time_reso).astype(np.int64)
    dds_file      = int(np.ceil(dds.max() / file_leng))
    block_file    = int(np.ceil(down_time_rate * block_size / file_leng))
    comb_file     = block_file + dds_file
    print(block_file, comb_file)
    print(f'file_list:{file_list}')

    ### loop
    for i in range(0, len(file_list), block_file):

        print('----加载文件阶段：---------------------')
        ### read data
        filename              = file_list[i].split('.fits')[0]
        print(f"    当前文件: {filename}")

        # 文件读取部分
        start_load_time = time.time()
        raw_data              = np.empty((0, 2, freq_reso))
        for j in range(comb_file):
            if i + j          < len(file_list):
                raw_data      = np.append(raw_data, load_fits_file(data_path + file_list[i + j]), axis=0)

        if Debug:
            # 输出 raw_data 的形状、内存占用和数据类型
            print(f"    读入的fits文件 形状: {raw_data.shape}")
            print(f"    读入的fits文件 内存占用: {raw_data.nbytes / (1024 ** 2):.2f} MB")  # 转换为 MB
            print(f"    读入的fits文件 数据类型: {raw_data.dtype}")
        
        
        # 在原始数据填充之前设置固定的随机种子
        RANDOM_SEED = 42  # 可以选择任何固定整数
        np.random.seed(RANDOM_SEED)

        if raw_data.shape[0]  < comb_file * file_leng:
            raw_data          = np.append(raw_data, np.random.rand(comb_file * file_leng - raw_data.shape[0], 2, freq_reso) * np.std(raw_data) + np.mean(raw_data), axis=0)
        raw_data              = np.mean(raw_data.reshape(comb_file * file_leng // down_time_rate, down_time_rate, 2, freq_reso//down_freq_rate, down_freq_rate), axis=(1, 2, 4)).astype(np.float32)
        # raw_data              = raw_data / np.mean(raw_data, axis=0)
        
        if Debug:
            # 输出 raw_data 的形状、内存占用和数据类型
            print(f"    当前文件的 形状: {raw_data.shape}")
            print(f"    当前文件的 内存占用: {raw_data.nbytes / (1024 ** 2):.2f} MB")  # 转换为 MB
            print(f"    当前文件的 数据类型: {raw_data.dtype}")


        end_load_time = time.time()
        print(f"加载文件耗时: {end_load_time - start_load_time:.2f} 秒")

        # CUDA计算部分
        start_time = time.time()
        print('----DM去色散数据处理阶段：---------------------')
        print(f"开始时间: {start_time:.2f} 秒")


        skip_DM = False
        if not skip_DM:
            new_data = d_dm_time_g(raw_data, height=DM_range, width=block_file*file_leng//down_time_rate)
        else:            
            print(f'$$$$$$$$$$$$$$ skip DM: $$$$$$$$$$$$$$')
            new_data = raw_data


        if loaded_data_flag:
            print(f'$$$$$$$$$$$$$$ load data: $$$$$$$$$$$$$$')
            # 加载数据
            loaded_data = load_data('./saved_data/new_data.npy')
            print(f"已加载的数据形状: {loaded_data.shape}")
            new_data = loaded_data
            
            # 检查数据是否正确加载
            if np.array_equal(new_data, loaded_data):
                print("数据加载和处理的数据 没有差别！")
            else:
                print("数据加载不正确！") 
        else:
            pass


        end_time = time.time()
        print(f"结束时间: {end_time:.2f} 秒") 

        # 计算耗时
        time_delay_correct = end_time - start_time
        print(f"----DM去色散数据处理阶段耗时: {time_delay_correct:.2f} 秒 -----------------")


        print(f"----模型推理数据阶段: 截止此时，对单个fits文件的 DM 去色散阶段已经全部完成。 -----------------")
        start_time = time.time()
        ### down_sampling and predict
        down_file_leng    = block_file * file_leng // down_time_rate
        data              = np.mean(new_data.reshape(3, DM_range // 2, 2, down_file_leng), axis=2).astype(np.float32)
        
        if Debug:
            print("Shape of infer data:", data.shape)
            # 输出 raw_data 的形状、内存占用和数据类型
            print(f"    当前用于推理的数据的 形状: {data.shape}")
            print(f"    当前用于推理的数据的 内存占用: {data.nbytes / (1024 ** 2):.2f} MB")  # 转换为 MB
            print(f"    当前用于推理的数据的 数据类型: {data.dtype}")


        print(down_file_leng // block_size)
        for j in range(down_file_leng // block_size):
            slice         = data[:, :, j * block_size: (j + 1) * block_size]

            # 在这里插入调用保存函数的代码: 
            npz_save_path = '/mnt/volume/userdata/zjt/dev/Projects/DRAFTS/Drafts-npu/processed_data_npuok'
            
            if npz_save_path:
                save_path_slice = save_processed_data(
                    slice_data=slice,
                    save_path=npz_save_path,
                    filename=filename,
                    j=j,
                    time_reso=time_reso,
                    down_time_rate=down_time_rate,
                    down_freq_rate=down_freq_rate,
                    DM_range=DM_range,
                    block_size=block_size,
                    verbose=Debug  # 使用Debug作为verbose标志
                )
                print(f"保存处理块到: {save_path_slice}")

            for k in range(3):
                img = preprocess_img(slice[k]).transpose([2, 0, 1])

                                
                # img 的 shape 
                print("Shape of img:", img.shape)

                save_file = Path (save_path) /  f'{filename}-TS{j:02d}-FS{k}_infer.jpg'
                save_infer_image = True
                if save_infer_image:  # 保存当前的image, 按照j k命令
                    infer_img = postprocess_img(img)  # 转回能画图的img 
                    plt.figure(figsize=(5, 4))
                    plt.imshow(infer_img, origin='lower')
                    plt.savefig(save_file, dpi=300, bbox_inches='tight')
                    plt.close()

                    # 输出保存的图片
                    print(f'Save image: {save_path}{filename}-TS{j}-FS{k}_infer.jpg')

                
                if j==6 and k==2:
                    if use_loaded_image:
                        loaded_img_data = load_data(f'./saved_data/img_j{j}k{k}.npy')
                        if np.array_equal(new_data, loaded_img_data):
                            print("img 数据加载正确！")
                        else:
                            print("img 数据加载不正确！")
                        new_data = loaded_img_data 

                        # 比较加载的npz数据和 这个数据是否一致。 
                        # j == 6 , block 6, k == 2. 
                        # 比较 npz 文件 和 原始数据 是否有区别。 
                        from compare_npy.load_npz import NPZUtils
                        file_path = '/mnt/volume/userdata/zjt/dev/Projects/DRAFTS/Drafts-npu/processed_data_npuok/M5_tracking_0011_block6.npz'
                        block6_data = NPZUtils.load_npz_file(file_path)
                        if block6_data :
                                for key, array in data.items():
                                    print(f"blocks 6 data 的数组 {key} 的形状为: {array.shape}")
                        # 比较 block_data[2] 和 slice[k]   , 这两个都是 block 6的 频率通道为3的 2d数组数据切片。



                with torch.no_grad():
                    hm, wh, offset = model(torch.from_numpy(img).to(f'npu:{device}' if isinstance(device, int) else device).float().unsqueeze(0))
                print('run inference done')

                if j==6 and k==2: 
                    if compare_error:
                        print('逐个比较推理结果和加载数据...') 
                        inference_data = [hm.cpu().numpy(), wh.cpu().numpy(), offset.cpu().numpy()] 
                        names = ["hm", "wh", "offset"]
                        
                        load_gpu_infer_result = []
                        # 循环比较
                        for name,  infer in zip(names, inference_data):
                            loaded_np_infer_data = load_data(f'./saved_data/img_j{j}k{k}_infer_{name}.npy')
                            load_gpu_infer_result.append(loaded_np_infer_data)
                            if np.array_equal(loaded_np_infer_data, infer):
                                print(f"{name} 数据一致！")
                            else:
                                print(f"{name} 数据不一致！")
    
                        loader_hm, loaded_wh, loaded_offset = load_gpu_infer_result[0], load_gpu_infer_result[1], load_gpu_infer_result[2]

                        save_data(hm, save_path='./saved_data/', filename='npu_infer_hm_fp16.npy')
                        save_data(wh, save_path='./saved_data/', filename='npu_infer_wh_fp16.npy')
                        save_data(offset, save_path='./saved_data/', filename='npu_infer_offset_fp16.npy')

                        ############################ 新增功能部分 ############################
                        # 输出hm最大值
                        print(f"\nhm最大值分析: Pred={hm.cpu().numpy().max():.4e}, GT={loader_hm.max():.4e}")


                        # 执行各维度分析
                        analyze_diff(hm.cpu().numpy(), loader_hm, "hm")
                        analyze_diff(wh.cpu().numpy(), loaded_wh, "wh") 
                        analyze_diff(offset.cpu().numpy(), loaded_offset, "offset")
                        #####################################################################

                        hm_error = calc_error(hm, loader_hm)
                        wh_error = calc_error(wh, loaded_wh)
                        offset_error = calc_error(offset, loaded_offset)

                        print(f'hm infer error: {hm_error}')
                        print(f'wh infer error: {wh_error}')
                        print(f'offset infer error: {offset_error}')
 
                    if change_load: 
                        # 替换 检测结果
                        hm, wh, offset = loader_hm, loaded_wh, loaded_offset
                        hm = torch.from_numpy(hm)
                        wh = torch.from_numpy(wh)
                        offset = torch.from_numpy(offset)
                    else:
                        pass


                use_npu = False
                start_time = time.time()
                top_conf, top_boxes = get_res(hm, wh, offset, confidence=det_prob)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f'检测结果处理阶段： get_res 函数执行时间: {elapsed_time:.6f} 秒, use_npu: {use_npu}')


                ## 画框并保存
                if top_boxes is not None:
                    print('drawing result picture.')
                    print('成功检测到目标，并绘制图像 原始和busrt 图像。')
                    img = postprocess_img(img) ## 转回能画图的img
                    for box in top_boxes:
                        left_x, left_y, right_x, right_y = box.astype(np.int64)
                        DM = (left_y + right_y) / 2 * (DM_range / 512)
                        dm_flag = True if DM > 20 else False
                        cv2.rectangle(img, (left_x, left_y), (right_x, right_y), (0, 220, 0), 1)
                        print(top_conf, DM)

                    if dm_flag:
                        # data_slice = new_data[k, :, j * block_size: (j + 1) * block_size]
                        # np.save('{}{}-TS{:0>2d}-FS{}.npy'.format(save_path, filename, j, k), data_slice.astype(np.float32))

                        TOA        = ((left_x + right_x) / 2 * (block_size / 512) + j * block_size) * down_time_rate * time_reso
                        toa_samp   = np.int64(TOA / time_reso / down_time_rate)
                        start_samp = np.max([0, toa_samp - 512])
                        freq_down  = np.mean(freq.reshape(freq_reso // down_freq_rate, down_freq_rate), axis=1)

                        dds        = np.int64(4.15 * DM * (freq_down ** -2 - freq_down.max() ** -2) * 1e3 / time_reso / down_time_rate)
                        burst      = raw_data[start_samp: start_samp + dds.max() + 2048, :]
                        new_data   = np.zeros((2048, 512))
                        for q in range(512):
                            new_data[:, q] = burst[dds[q]: dds[q] + 2048, q]
                        new_data   = np.mean(new_data.reshape(512, 4, 512, 1), axis=(1, 3))
                        new_data   = new_data / np.mean(new_data, axis=0)
                        vmin, vmax = np.percentile(new_data, [5, 95])
                        


                        plt.figure(figsize=(5, 4))
                        plt.imshow(new_data.T, aspect='auto', origin='lower', cmap='mako', vmin=vmin, vmax=vmax)
                        plt.yticks(np.linspace(0, 512, 6), np.round(np.linspace(freq.min(), freq.max(), 6)).astype(np.int64))
                        plt.xticks(np.linspace(0, 512, 5), np.round(np.linspace(0, 512, 5)*time_reso*down_time_rate*4*1e3, 2))
                        plt.xlabel('Time (ms)')
                        plt.ylabel('Frequency (MHz)')
                        
                        save_file = Path (save_path) /  f'{filename}-TS{j:02d}-FS{k}-Burst.jpg'
                        plt.savefig(save_file, dpi=300, bbox_inches='tight')
                        plt.show()

                        plt.figure(figsize=(5, 4))
                        plt.imshow(img, origin='lower')
                        save_file2 = Path (save_path) /  f'{filename}-TS{j:02d}-FS{k}.jpg'
                        plt.savefig(save_file2, dpi=300, bbox_inches='tight')
                        plt.close()
        del new_data
        # 记录结束时间
        end_time = time.time()
        # 计算总耗时
        infer_total_time = end_time - start_time
        print(f"----模型推理数据阶段总耗时: {infer_total_time:.2f} 秒-----------------")

     
    all_end_time = time.time()
    # 计算总耗时
    total_time = all_end_time - all_start_time
    print(f"$$$$ 程序总耗时: {total_time:.2f} 秒. $$$$")

    # profiler.disable()
    
    # print(f"$$$$ profile分析结果: $$$$")
    # stats = pstats.Stats(profiler).sort_stats('cumtime').print_stats(20)
    # snakeviz.pstats2html(pstats.Stats(profiler), 'profile.html')
