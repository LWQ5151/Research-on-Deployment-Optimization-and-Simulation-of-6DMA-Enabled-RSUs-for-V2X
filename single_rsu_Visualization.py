import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge  # 用于绘制扇形
# [V26] 移除 LogNorm
# from matplotlib.colors import LogNorm
import os
import time
import math
import matplotlib.ticker as mticker
# [V27] 移除 V21 依赖
# from itertools import combinations

# --- [ 字体配置 ] ---
import matplotlib

try:
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
except Exception as e:
    print(f"Warning: Could not set preferred font. Using default. Error: {e}")
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams.update({
    'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 14,  # 增大了字体
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'figure.titlesize': 18
})
# --- [ 字体配置结束 ] ---

# --- 1. 配置参数 ---

# (要分析的 RSU 方案)
DEPLOYMENT_STRATEGY_NAME = 'dps_greedy'

# (输入文件)
DEPLOYMENT_FILE = f'deploy_results/deployment_{DEPLOYMENT_STRATEGY_NAME}.csv'
TRAJ_VOLUME_FILE = 'data/traj_volume.npy'  # 5分钟粒度的 3D 网格数据
ANGULAR_HIST_FILE = f'results/angular_histograms_{DEPLOYMENT_STRATEGY_NAME}.npy'  # 5分钟粒度的 3D 角度数据
METADATA_FILE = 'data/grid_metadata.npy'

# (输出文件)
OUTPUT_DIR = 'results/snapshots/'
# [V27] 更新文件名
OUTPUT_PLOT_FILE = os.path.join(OUTPUT_DIR, f'beam_steering_snapshots_V19_Greedy_{DEPLOYMENT_STRATEGY_NAME}_RSU1.pdf')

# (天线参数)
BEAM_WIDTH_DEGREES = 60
NUM_BEAMS = 3
RSU_COVERAGE_RADIUS_METERS = 500.0
EPSILON = 1e-9

INPUT_TIME_WINDOW_MINUTES = 5

# (可视化参数)
TIMESLOTS_TO_PLOT = [36, 90, 102, 144, 210, 222, 252, 276]
CROP_RADIUS = 30
NUM_ANGLES = 360


# [V27] 移除 V21 参数
# TOP_N_CANDIDATES = 20

# --- 2. 核心算法函数 (V19 - 贪心) ---
def find_all_beam_scores(histogram, beam_width_deg):
    """(V19) 返回所有 360 个起始角度的得分"""
    padded_hist = np.concatenate((histogram, histogram))
    cum_sum = np.cumsum(padded_hist)
    cum_sum_padded = np.insert(cum_sum, 0, 0)
    scores = cum_sum_padded[beam_width_deg: beam_width_deg + NUM_ANGLES] - \
             cum_sum_padded[0: NUM_ANGLES]
    return scores


def calculate_coverage_score(histogram, angles, beam_width_deg):
    """(V19) 计算给定起始角度的最终覆盖分数 (并集)"""
    coverage_mask = np.zeros(NUM_ANGLES, dtype=bool)
    for start_angle_in in angles:
        if start_angle_in is None or np.isnan(start_angle_in): continue
        start_angle = int(np.round(start_angle_in))
        end_angle = (start_angle + beam_width_deg - 1) % NUM_ANGLES
        if start_angle <= end_angle:
            coverage_mask[start_angle: end_angle + 1] = True
        else:
            coverage_mask[start_angle: NUM_ANGLES] = True
            coverage_mask[0: end_angle + 1] = True
    return np.sum(histogram[coverage_mask])


def find_best_beam_greedy(histogram, beam_width_deg):
    """(V19) 查找单个最佳波束的起始角度和分数"""
    scores = find_all_beam_scores(histogram, beam_width_deg)
    best_start_angle = np.argmax(scores)
    max_score = scores[best_start_angle]
    return best_start_angle, max_score


def mask_histogram(histogram, start_angle_in, beam_width_deg):
    """(V19) 安全地掩码一个波束"""
    hist_copy = histogram.copy()
    if start_angle_in is None or np.isnan(start_angle_in): return hist_copy
    start_angle = int(np.round(start_angle_in))
    end_angle = (start_angle + beam_width_deg - 1) % NUM_ANGLES
    if start_angle <= end_angle:
        hist_copy[start_angle: end_angle + 1] = 0
    else:
        hist_copy[start_angle: NUM_ANGLES] = 0
        hist_copy[0: end_angle + 1] = 0
    return hist_copy


def run_sas_optimization(histogram, num_beams, beam_width_deg):
    """
    (V19 修正版) 运行 *贪心* SAS 算法
    [修改] 这是唯一的 SAS 优化器
    """
    angles = []
    hist_to_optimize = histogram.copy()
    for _ in range(num_beams):
        best_angle, score = find_best_beam_greedy(hist_to_optimize, beam_width_deg)
        if score < EPSILON:
            angles.append(np.nan)
        else:
            angles.append(best_angle)
            if best_angle is not None and not np.isnan(best_angle):
                hist_to_optimize = mask_histogram(hist_to_optimize, best_angle, beam_width_deg)
    while len(angles) < num_beams:
        angles.append(np.nan)
    return angles


# [V27] 移除 V21 优化器
# def run_sas_optimization_v21(...):
#     ...

# --- 3. 主可视化流程 ---

def main():
    # 0. 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载所有数据
    print("Loading all required data...")
    try:
        rsus_df = pd.read_csv(DEPLOYMENT_FILE)
        metadata = np.load(METADATA_FILE, allow_pickle=True).item()
        traj_volume_stacked = np.load(TRAJ_VOLUME_FILE)  # (nx, ny, nt) -> (i, j, k)
        angular_histograms_5min = np.load(ANGULAR_HIST_FILE)
    except FileNotFoundError as e:
        print(f"Error: Cannot load input file {e.filename}.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. 获取 RSU #1 (或 ID=1) 的信息
    try:
        rsu_to_plot = rsus_df[rsus_df['id'] == 1].iloc[0]
        rsu_hist_index = rsus_df[rsus_df['id'] == 1].index[0]
        i_center = int(rsu_to_plot['grid_i'])  # 绝对 X 索引
        j_center = int(rsu_to_plot['grid_j'])  # 绝对 Y 索引
        print(
            f"Target RSU: ID {rsu_to_plot['id']} at grid (i={i_center}, j={j_center}) using hist index {rsu_hist_index}")
    except (IndexError, KeyError):
        print(f"Error: Cannot find RSU ID=1 in {DEPLOYMENT_FILE}.")
        return

    # 3. 准备参数
    grid_size = metadata['grid_size_meters']
    radius_in_grids = RSU_COVERAGE_RADIUS_METERS / grid_size  # 500/25 = 20
    nx, ny, nt = traj_volume_stacked.shape  # nx=X_dim, ny=Y_dim

    if angular_histograms_5min.shape[0] < (rsu_hist_index + 1) or \
            angular_histograms_5min.shape[1] != nt:
        print("Error: Angular histogram file mismatch (RSU count or timeslots).")
        return

    # 4. 创建 2x4 图窗
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5))
    axes_flat = axes.flatten()

    print("Generating 8 snapshots using V19 (Greedy) Optimizer...")  # [V27] 修改

    # [V26] 查找所有时隙的 99 百分位 vmax
    all_vmax_local = []
    for k in TIMESLOTS_TO_PLOT:
        i_min_abs = max(0, i_center - CROP_RADIUS);
        i_max_abs = min(nx, i_center + CROP_RADIUS)
        j_min_abs = max(0, j_center - CROP_RADIUS);
        j_max_abs = min(ny, j_center + CROP_RADIUS)
        cropped_traffic_map = traj_volume_stacked[i_min_abs:i_max_abs, j_min_abs:j_max_abs, k]
        if cropped_traffic_map.size > 0:
            all_vmax_local.append(np.max(cropped_traffic_map))
    global_vmax_percentile = np.percentile(all_vmax_local, 99.5) if all_vmax_local else 100  # 提高到 99.5%
    if global_vmax_percentile == 0: global_vmax_percentile = 1
    print(f"Using Linear scale with unified 99.5th percentile vmax: {global_vmax_percentile:.2f}")

    # 5. 循环绘制 8 个时隙
    for i, k in enumerate(TIMESLOTS_TO_PLOT):
        ax = axes_flat[i]

        hist_k = angular_histograms_5min[rsu_hist_index, k, :]
        # [V27] 使用 V19 贪心算法
        oracle_angles_start = run_sas_optimization(hist_k, NUM_BEAMS, BEAM_WIDTH_DEGREES)

        # [V26 坐标修正] i=x, j=y
        i_min_abs = max(0, i_center - CROP_RADIUS);
        i_max_abs = min(nx, i_center + CROP_RADIUS)  # X 范围
        j_min_abs = max(0, j_center - CROP_RADIUS);
        j_max_abs = min(ny, j_center + CROP_RADIUS)  # Y 范围

        plot_i_min = i_center - CROP_RADIUS;
        plot_i_max = i_center + CROP_RADIUS  # X 轴限制
        #plot_j_min = j_center - CROP_RADIUS;
        plot_j_min = 0
        plot_j_max = j_center + CROP_RADIUS  # Y 轴限制

        # 提取数据 (x_slice, y_slice)
        cropped_traffic_map = traj_volume_stacked[i_min_abs:i_max_abs, j_min_abs:j_max_abs, k]

        # c. 绘制背景热力图
        vmin = 0  # 线性刻度从 0 开始

        plot_grid = np.zeros((nx, ny))  # (x_dim, y_dim)
        plot_grid[i_min_abs:i_max_abs, j_min_abs:j_max_abs] = cropped_traffic_map

        # [V26 修正]
        # imshow(data[row, col]) -> data[y_dim, x_dim]
        # 我们有 plot_grid[x_dim, y_dim]，所以我们绘制 plot_grid.T
        # extent=[x_min, x_max, y_min, y_max]
        im = ax.imshow(plot_grid.T, origin='lower', cmap='viridis',  # [V26] 转置并改回 viridis
                       extent=[0, nx, 0, ny],  # 完整范围 (x_min, x_max, y_min, y_max)
                       interpolation='nearest',
                       norm=plt.Normalize(vmin=vmin, vmax=global_vmax_percentile),  # 线性刻度
                       zorder=1,
                       alpha=1.0
                       )

        # e. 绘制波束
        for start_angle in oracle_angles_start:
            if np.isnan(start_angle): continue
            end_angle = start_angle + BEAM_WIDTH_DEGREES

            # [V26 修正] Wedge(center=(x, y), ...)
            wedge = Wedge(
                (i_center, j_center),  # RSU 绝对坐标 (x=i, y=j)
                radius_in_grids,
                start_angle,
                end_angle,
                facecolor='none',
                edgecolor='r',  # [V26] 红色, 在 viridis 上更清晰
                alpha=0.9,
                linewidth=1.5,
                linestyle='--',
                zorder=2
            )
            ax.add_patch(wedge)

        # f. 绘制 RSU 中心点
        # [V26 修正] plot(x, y)
        ax.plot(i_center, j_center, 'w+', markersize=10, markeredgewidth=2, label='RSU Center', zorder=3)

        # g. 设置坐标轴和标题 (英文)
        hour = (k * INPUT_TIME_WINDOW_MINUTES) // 60
        minute = (k * INPUT_TIME_WINDOW_MINUTES) % 60
        ax.set_title(f"Timeslot k={k} ({hour:02d}:{minute:02d})")

        # [V26 修正] xlim(x_min, x_max), ylim(y_min, y_max)
        ax.set_xlim(plot_i_min, plot_i_max)
        ax.set_ylim(plot_j_min, plot_j_max)

        ax.set_xlabel("Grid Index (i-axis)")
        ax.set_ylabel("Grid Index (j-axis)")
        ax.set_aspect('equal')

    # 6. [V25] 为所有子图添加一个统一的色条
    # [V26] 修复 cbar 的 ax 参数
    #fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.03, label='GPS Points (Linear Scale)')

    # 7. 调整布局并保存
    # [V27] 修改标题
    fig.suptitle(
        f'Dynamic Beam Steering Snapshots for RSU ID {rsu_to_plot["id"]} (Strategy C: Oracle - V19 Greedy Optimizer)',
        fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.0)

    try:
        plt.savefig(OUTPUT_PLOT_FILE, format='pdf', dpi=300)
        print(f"\n--- Visualization successful! ---")
        print(f"Snapshots saved to: {OUTPUT_PLOT_FILE}")
    except Exception as e:
        print(f"\nError saving plot: {e}")

    plt.show()


if __name__ == "__main__":
    main()

