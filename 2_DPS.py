import numpy as np
from scipy.stats import entropy
from scipy.signal import convolve2d
import time
import matplotlib.pyplot as plt
from skimage.util import view_as_windows


# --- 1. 常量与辅助函数 ---

# 定义时隙总数 (e.g., 24h * 12 per hour = 288)
NUM_TIMESLOTS = 288
# 卷积核大小 (5x5)
KERNEL_SIZE = 9
# 小的常数，防止除以0
EPSILON = 1e-9


def create_gaussian_kernel(size=5, sigma=1.0):
    """
    创建一个归一化的高斯权重核 (总和为1)
    如您所述，距离中心越近，权重越高。
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    # 归一化
    #kernel = kernel / np.sum(kernel)
    return kernel

def calculate_similarity(center, neighbor):
    """
    向量化计算: Sim = 1 - (AE / max(|a|, |b|))
    """
    Absolute_Error = np.abs(center - neighbor)
    Max_Magnitude = np.maximum(center, neighbor)
    # Max_Magnitude >= 1 (因为我们用了 C+1)，所以除法是安全的
    return 1.0 - (Absolute_Error / Max_Magnitude)


def similarity_based_convolution(grid, W, kernel_size=9):
    k = kernel_size // 2  # 4（9x9的中心索引为(4,4)）

    # 步骤1：提取所有9x9滑动窗口（边界处理：只保留完整窗口，输出尺寸会缩小）
    # 输出shape：(new_h, new_w, 9, 9)，其中new_h=338-9+1=330，new_w=329-9+1=321
    windows = view_as_windows(grid, window_shape=(kernel_size, kernel_size))

    # 步骤2：提取每个窗口的中心点值（shape：(new_h, new_w)）
    center_vals = windows[:, :, k, k]  # 每个窗口的(4,4)位置是中心点

    # 将 uint32 转换为 int64，以安全地执行减法
    windows = windows.astype(np.int64)
    center_vals = center_vals.astype(np.int64)

    # 步骤3：批量计算所有窗口的相似度矩阵S
    # 窗口形状：(new_h, new_w, 9, 9)，中心点扩展为(new_h, new_w, 1, 1)以广播
    # 相似度函数用你之前的标量相似度（这里用相对误差转换，可替换为你的函数）
    abs_diff = np.abs(windows - center_vals[:, :, np.newaxis, np.newaxis])  # 每个元素与中心的绝对差
    max_vals = np.maximum(np.abs(windows), np.abs(center_vals[:, :, np.newaxis, np.newaxis]))  # 避免除0
    max_vals = np.where(max_vals == 0, 1e-8, max_vals)  # 处理中心值为0的情况
    S = 1 - (abs_diff / max_vals)  # 相似度矩阵（shape：(new_h, new_w, 9, 9)）

    # 步骤4：S与W逐元素相乘后求和（完成卷积）
    # W扩展为(1,1,9,9)以广播，最后两个维度求和得到每个窗口的结果
    result = np.sum(S * W[np.newaxis, np.newaxis, :, :], axis=(-2, -1))

    return result


# --- 2. 主计算函数 ---

def calculate_potential_heatmap(C):
    """
    根据您定义的逻辑，计算潜力热力图
    C: 3D 数据立方体 TrajVolume[nx, ny, nt]
    """

    # 获取维度
    if C.ndim != 3 or C.shape[2] != NUM_TIMESLOTS:
        raise ValueError(f"输入数据立方体 C 必须是 (nx, ny, {NUM_TIMESLOTS}) 维度")
    nx, ny, nt = C.shape
    print(f"开始计算... 网格维度: {nx}x{ny}, 时间窗口: {nt}")

    # --- 1. 计算时间波动性 (S_temporal_cv) ---
    print("步骤 1/3: 计算时间波动性 (Coefficient of Variation)...")
    start_time = time.time()

    # 计算每个网格的96维GPS点总和
    # (nx, ny)
    C_sum_t = C.sum(axis=2)


    # 计算每个网格的时间标准差 (nx, ny)
    S_temporal_cv = C.std(axis=2)

    print(f"  ...时间波动性计算完成，耗时: {time.time() - start_time:.2f} 秒")


    # --- 2. 计算分子: 空间分 S_spatial ---
    print("步骤 2/3: 计算空间分 (S_spatial)...")
    start_time = time.time()

    # 2a. 计算幅度因子 S_mag (nx, ny)
    # 获取全局最大值
    C_max = C.max()
    if C_max == 0:
        print("警告: 数据立方体为空，所有潜力分将为 0。")
        # 提前返回 0 矩阵
        return np.zeros((nx, ny))

    # 计算每个网格的时间平均GPS点数
    C_mean_t = C.mean(axis=2)
    # 归一化幅度因子
    S_mag = C_mean_t / C_mean_t.max()

    # 2b. 计算平均相对结构分 S_rel_bar (nx, ny)

    # 创建高斯核 (5x5)
    W = create_gaussian_kernel(size=KERNEL_SIZE, sigma=4.0)

    # 准备 C+1 数据
    C_plus1 = C + 1

    # 初始化一个 (nx, ny, nt) 的 3D 数组来存储每个时隙的 S_rel_k
    S_rel_k_cube = np.zeros_like(C, dtype=float)

    # KERNEL_SIZE=9, hf=4. m,n from -4 to +4
    hf = KERNEL_SIZE // 2


    print("  ...正在计算时隙内空间相似度 (S_rel_k)...")
    # 遍历所有 96 个时间切片
    for k in range(nt):
        # 提取当前时间切片 (nx, ny)
        Slice_plus1_k = C[:, :, k]

        # 1. 对邻居进行加权求和
        #边缘补0
        padded_grid = np.pad(
            Slice_plus1_k,
            pad_width=hf,  # 上下左右各补k行/列
            mode='constant'  # 根据需求选择填充模式
        )

        convolved_similarity_k = similarity_based_convolution(padded_grid, W, KERNEL_SIZE)

        # 5. 存储该时隙的相似度
        S_rel_k_cube[:, :, k] = convolved_similarity_k
        # **********************************************


    # 计算时间上的平均值 (nx, ny)
    S_rel_bar = S_rel_k_cube.mean(axis=2)

    # 2c. 组合得到最终空间分
    S_spatial = S_rel_bar * S_mag




    print(f"  ...空间分计算完成，耗时: {time.time() - start_time:.2f} 秒")

    # --- 3. 计算最终潜力分 Score ---
    print("步骤 3/3: 计算最终潜力分...")

    Score = S_spatial * S_temporal_cv

    # 对于时间总和为0的网格，其 S_spatial 和 S_temporal 都会是0，
    # 最终 Score 也应为0 (0/EPSILON ~= 0)
    Score[C_sum_t == 0] = 0

    # (可选) 归一化到 [0, 1] 用于可视化
    if Score.max() > 0:
        Score_normalized = (Score - Score.min()) / (Score.max() - Score.min())
    else:
        Score_normalized = Score

    print("--- 全部计算完成 ---")

    return Score, Score_normalized


# --- 3. 示例执行 ---

if __name__ == "__main__":
    # 假设我们已经有了 'traj_volume.npy'
    # 这里我们生成一个假的 TrajVolume 来测试

    # 1. 生成模拟数据 (请替换为您的真实数据加载)
    try:
        # 尝试加载真实数据
        C = np.load('data/traj_volume.npy')
        print(f"成功加载 'traj_volume.npy', 形状: {C.shape}")
    except FileNotFoundError:
        print("未找到 'traj_volume.npy'，将使用模拟数据进行测试...")
        NX, NY = 100, 100  # 100x100 的网格
        NT = NUM_TIMESLOTS  # 96 个时隙
        C = np.zeros((NX, NY, NT))

        # 创建一个 "空间聚集" & "时间尖峰" 的高潜力点 (30, 30)
        # "时间尖峰": 只在 k=10 和 k=70 两个时隙有流量
        # "空间聚集": 流量集中在 (30, 30) 及其近邻
        idx, idy = 30, 30
        for i in range(-1, 2):  # 3x3 区域
            for j in range(-1, 2):
                C[idx + i, idy + j, 10] = 50 * (1 - (abs(i) + abs(j)) / 2.0)  # 早高峰
                C[idx + i, idy + j, 70] = 60 * (1 - (abs(i) + abs(j)) / 2.0)  # 晚高峰

        # 创建一个 "空间聚集" & "时间平稳" 的低潜力点 (70, 70)
        # "时间平稳": 流量均匀分布在所有时隙
        idx, idy = 70, 70
        for k in range(NT):
            C[idx, idy, k] = 2  # 中心点流量平稳
            C[idx + 1, idy, k] = 1  # 邻居流量平稳

    # 2. 执行计算
    Score_raw, Score_norm = calculate_potential_heatmap(C)

    # 3. 保存结果
    np.save('data/potential_score_raw.npy', Score_raw)
    np.save('data/potential_score_normalized.npy', Score_norm)

    print("\n计算结果已保存：")
    print("  - 'potential_score_raw.npy' (原始分值)")
    print("  - 'potential_score_normalized.npy' (归一化分值，用于可视化)")

    # (可选) 打印一些统计信息
    print(f"\n原始分数统计: min={Score_raw.min():.2f}, max={Score_raw.max():.2f}, mean={Score_raw.mean():.2f}")

    # 验证模拟数据
    if 'NX' in locals():
        print(f"模拟点 (30,30) 潜力分: {Score_raw[30, 30]:.4f} (应为高分)")
        print(f"模拟点 (70,70) 潜力分: {Score_raw[70, 70]:.4f} (应为低分)")
        print(f"空区域 (10,10) 潜力分: {Score_raw[10, 10]:.4f} (应为 0)")