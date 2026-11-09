import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import math
import matplotlib.ticker as mticker
# [V19] 移除 V21 算法所依赖的 combinations
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
    'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 10,  # 减小图例字体
    'figure.titlesize': 18
})
# --- [ 字体配置结束 ] ---

# --- 1. 配置参数 ---

# (输出目录)
RESULTS_DIR = 'results/synthetic_experiment/'
# [修改] 为两个场景定义不同的直方图文件
OUTPUT_HIST_FILE_ORIGINAL = os.path.join(RESULTS_DIR, 'angular_histograms_synthetic_original.npy')
OUTPUT_HIST_FILE_HIGHSIGNAL = os.path.join(RESULTS_DIR, 'angular_histograms_synthetic_highsignal.npy')
OUTPUT_RESULTS_CSV = os.path.join(RESULTS_DIR, 'synthetic_evaluation_results.csv')
OUTPUT_PLOT_FILE = os.path.join(RESULTS_DIR, 'synthetic_evaluation_comparison_plot.pdf')  # 2x2 图表

# (天线参数)
BEAM_WIDTH_DEGREES = 60
NUM_ANGLES = 360
NUM_BEAMS = 3
EPSILON = 1e-9

# (时间参数)
INPUT_TIME_WINDOW_MINUTES = 5
INPUT_NUM_TIMESLOTS = (24 * 60) // INPUT_TIME_WINDOW_MINUTES  # 288

# (人造数据参数 - 场景 1: 原始)
PEAK_TRAFFIC_ORIGINAL = 500
MIDDAY_NOISE_ORIGINAL = 50
NIGHT_NOISE_ORIGINAL = 3

# (人造数据参数 - 场景 2: 高信噪比)
PEAK_TRAFFIC_HIGH = 2000  # [修改] 大幅提高峰值
MIDDAY_NOISE_HIGH = 5  # [修改] 大幅降低噪声
NIGHT_NOISE_HIGH = 1  # [修改] 大幅降低噪声


# --- 2. 核心算法函数 (V19) ---
# (find_all_beam_scores, find_best_beam_greedy, calculate_coverage_score,
#  mask_histogram, run_sas_greedy_v19 保持不变)
def find_all_beam_scores(histogram, beam_width_deg):
    """返回所有 360 个起始角度的得分"""
    padded_hist = np.concatenate((histogram, histogram))
    cum_sum = np.cumsum(padded_hist)
    cum_sum_padded = np.insert(cum_sum, 0, 0)
    scores = cum_sum_padded[beam_width_deg: beam_width_deg + NUM_ANGLES] - \
             cum_sum_padded[0: NUM_ANGLES]
    return scores


def find_best_beam_greedy(histogram, beam_width_deg):
    """(V19) 查找单个最佳波束的起始角度和分数 (贪心算法用)"""
    scores = find_all_beam_scores(histogram, beam_width_deg)
    best_start_angle = np.argmax(scores)
    max_score = scores[best_start_angle]
    return best_start_angle, max_score


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


def run_sas_greedy_v19(histogram, num_beams, beam_width_deg):
    """(V19) 运行 *贪心* SAS 算法"""
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


# --- 3. 人造数据生成函数 (参数化) ---

def generate_synthetic_histogram(output_file, peak_traffic, midday_noise, night_noise):
    """
    [修改] 生成参数化的人造数据
    返回: (1, 288, 360) 形状的 NumPy 数组
    """
    print(f"Generating synthetic histogram for: {os.path.basename(output_file)}")
    print(f"  Params: PEAK={peak_traffic}, MIDDAY={midday_noise}, NIGHT={night_noise}")
    hist = np.zeros((1, INPUT_NUM_TIMESLOTS, NUM_ANGLES), dtype=np.uint32)

    # 时段 1: 深夜 (00:00 - 06:00), k = 0..71 (72 个时隙)
    hist[0, 0:72, :] = np.random.randint(0, night_noise + 1, size=(72, NUM_ANGLES))

    # 时段 2: 早高峰 (06:00 - 10:00), k = 72..119 (48 个时隙)
    hist[0, 72:120, :] = np.random.randint(0, night_noise + 1, size=(48, NUM_ANGLES))
    hist[0, 72:120, 90] = peak_traffic
    hist[0, 72:120, 270] = peak_traffic

    # 时段 3: 平峰 (10:00 - 17:00), k = 120..203 (84 个时隙)
    hist[0, 120:204, :] = np.random.randint(0, midday_noise + 1, size=(84, NUM_ANGLES))

    # 时段 4: 晚高峰 (17:00 - 21:00), k = 204..251 (48 个时隙)
    hist[0, 204:252, :] = np.random.randint(0, night_noise + 1, size=(48, NUM_ANGLES))
    hist[0, 204:252, 0] = peak_traffic
    hist[0, 204:252, 180] = peak_traffic

    # 时段 5: 深夜 (21:00 - 24:00), k = 252..287 (36 个时隙)
    hist[0, 252:288, :] = np.random.randint(0, night_noise + 1, size=(36, NUM_ANGLES))

    print(f"  Synthetic histogram generated. Total GPS points: {np.sum(hist):,}")

    # 保存文件
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(output_file, hist)
    print(f"  Synthetic histogram saved to: {output_file}")

    return hist


# --- 4. 核心评估流程 (重构为函数) ---

def run_full_evaluation(hist_5min, sas_solver):
    """
    对给定的直方图数据运行 A/B/C 评估
    返回: (total_score_A, total_score_B, total_score_C)
    """
    print("  Calculating Static-Optimal (A) angles...")
    hist_total_day = np.sum(hist_5min[0, :, :], axis=0)  # [0] 因为只有一个 RSU
    static_angles = sas_solver(hist_total_day)
    print(f"    Static angles found: {static_angles}")

    print("  Calculating Predictive-Dynamic (B) angles for all timeslots...")
    dynamic_angles_predictive = np.zeros((INPUT_NUM_TIMESLOTS, NUM_BEAMS))
    for k in range(INPUT_NUM_TIMESLOTS):
        k_prev = (k - 1 + INPUT_NUM_TIMESLOTS) % INPUT_NUM_TIMESLOTS
        hist_k_prev = hist_5min[0, k_prev, :]
        dynamic_angles_predictive[k, :] = sas_solver(hist_k_prev)

    print("  Running A/B/C evaluation loop...")
    total_score_A = 0
    total_score_B = 0
    total_score_C = 0

    for k_5min in range(INPUT_NUM_TIMESLOTS):
        hist_k_5min = hist_5min[0, k_5min, :]  # 真实数据

        total_score_A += calculate_coverage_score(hist_k_5min, static_angles, BEAM_WIDTH_DEGREES)
        angles_B = dynamic_angles_predictive[k_5min, :]
        total_score_B += calculate_coverage_score(hist_k_5min, angles_B, BEAM_WIDTH_DEGREES)
        angles_C = sas_solver(hist_k_5min)
        total_score_C += calculate_coverage_score(hist_k_5min, angles_C, BEAM_WIDTH_DEGREES)

    print("  Evaluation loop complete.")
    return total_score_A, total_score_B, total_score_C


# --- 5. 新增绘图函数 ---

def plot_data_distribution(ax, histogram_data, title):
    """
    绘制 (1, 288, 360) 直方图的数据分布热力图
    """
    print(f"  Plotting distribution for: {title}")
    # 移除 RSU 维度，并转置为 (Angle, Time) 以便 imshow
    hist_2d = histogram_data[0, :, :].T  # Shape (360, 288)

    # 使用 LogNorm 增强非高峰时段的可见性
    vmin = np.min(hist_2d[hist_2d > 0]) if np.any(hist_2d > 0) else 1
    im = ax.imshow(hist_2d, aspect='auto', interpolation='nearest', origin='lower',
                   cmap='viridis', norm=plt.Normalize(vmin=0, vmax=np.max(hist_2d) * 0.5))  # 线性，但压低vmax
    # norm=LogNorm(vmin=vmin, vmax=np.max(hist_2d))) # 对数

    ax.set_title(title)
    ax.set_xlabel('Timeslot (5-min intervals)')
    ax.set_ylabel('Angle (Degrees)')
    ax.set_yticks([0, 90, 180, 270, 359])
    return im


def plot_abc_comparison(ax, scores, title):
    """
    绘制 A/B/C 对比条形图
    """
    print(f"  Plotting A/B/C comparison for: {title}")
    labels = ['A: Static-Optimal', 'B: Predictive-Dynamic', 'C: Oracle-Dynamic']
    colors = ['b', 'orange', 'g']

    bars = ax.bar(labels, scores, color=colors, alpha=0.8)

    ax.set_ylabel('Total GPS Points Covered')
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 自动调整 Y 轴限制
    ax.set_ylim(bottom=0, top=max(scores) * 1.15)

    # 在条形上添加数值标签
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:,.0f}',
                ha='center', va='bottom', fontsize=10)


# --- 6. 主评估流程 (V22 - 双实验对比) ---

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sas_solver = lambda h: run_sas_greedy_v19(h, NUM_BEAMS, BEAM_WIDTH_DEGREES)
    print("--- Using V19 (Greedy) SAS Optimizer ---")

    # --- 实验 1: 原始场景 ---
    print("\n--- Experiment 1: Original Scenario ---")
    if os.path.exists(OUTPUT_HIST_FILE_ORIGINAL):
        print(f"Loading existing histogram: {OUTPUT_HIST_FILE_ORIGINAL}")
        hist_1 = np.load(OUTPUT_HIST_FILE_ORIGINAL)
    else:
        hist_1 = generate_synthetic_histogram(
            OUTPUT_HIST_FILE_ORIGINAL,
            PEAK_TRAFFIC_ORIGINAL,
            MIDDAY_NOISE_ORIGINAL,
            NIGHT_NOISE_ORIGINAL
        )
    (score_A1, score_B1, score_C1) = run_full_evaluation(hist_1, sas_solver)

    # --- 实验 2: 高信噪比场景 ---
    print("\n--- Experiment 2: High-Signal Scenario ---")
    if os.path.exists(OUTPUT_HIST_FILE_HIGHSIGNAL):
        print(f"Loading existing histogram: {OUTPUT_HIST_FILE_HIGHSIGNAL}")
        hist_2 = np.load(OUTPUT_HIST_FILE_HIGHSIGNAL)
    else:
        hist_2 = generate_synthetic_histogram(
            OUTPUT_HIST_FILE_HIGHSIGNAL,
            PEAK_TRAFFIC_HIGH,
            MIDDAY_NOISE_HIGH,
            NIGHT_NOISE_HIGH
        )
    (score_A2, score_B2, score_C2) = run_full_evaluation(hist_2, sas_solver)

    # --- 7. 整理并保存结果 ---
    results_data = [
        {
            'Scenario': 'Original (500 Peak, 50 Noise)',
            'Score_A_Static': score_A1,
            'Score_B_Predictive': score_B1,
            'Score_C_Oracle': score_C1,
            'PotentialGain_Pct ((C-A)/A)': (score_C1 / (score_A1 + EPSILON) - 1) * 100,
            'RealizedGain_Pct ((B-A)/A)': (score_B1 / (score_A1 + EPSILON) - 1) * 100,
        },
        {
            'Scenario': 'High-Signal (2000 Peak, 5 Noise)',
            'Score_A_Static': score_A2,
            'Score_B_Predictive': score_B2,
            'Score_C_Oracle': score_C2,
            'PotentialGain_Pct ((C-A)/A)': (score_C2 / (score_A2 + EPSILON) - 1) * 100,
            'RealizedGain_Pct ((B-A)/A)': (score_B2 / (score_A2 + EPSILON) - 1) * 100,
        }
    ]
    results_df = pd.DataFrame(results_data)
    try:
        results_df.to_csv(OUTPUT_RESULTS_CSV, index=False, float_format='%.2f')
        print(f"\nSynthetic evaluation results saved to: {OUTPUT_RESULTS_CSV}")
    except Exception as e:
        print(f"\nError saving results CSV: {e}")

    # 8. 打印全局总结
    print("\n--- Synthetic Scenarios Performance Summary ---")
    print(results_df.to_string(index=False))

    # 9. 可视化 (2x2)
    print(f"\nGenerating 2x2 comparison plot to {OUTPUT_PLOT_FILE}...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))  # 2x2 图表

    # 绘制左侧 (原始场景)
    im1 = plot_data_distribution(axes[0, 0], hist_1, 'Data Distribution (Original Scenario)')
    plot_abc_comparison(axes[1, 0], [score_A1, score_B1, score_C1], 'A/B/C Performance (Original Scenario)')

    # 绘制右侧 (高信噪比场景)
    im2 = plot_data_distribution(axes[0, 1], hist_2, 'Data Distribution (High-Signal Scenario)')
    plot_abc_comparison(axes[1, 1], [score_A2, score_B2, score_C2], 'A/B/C Performance (High-Signal Scenario)')

    # 添加色条
    fig.colorbar(im1, ax=axes[0, 0], orientation='vertical', fraction=0.046, pad=0.04,
                 label='GPS Points (Linear Scale)')
    fig.colorbar(im2, ax=axes[0, 1], orientation='vertical', fraction=0.046, pad=0.04,
                 label='GPS Points (Linear Scale)')

    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 留出总标题空间
    fig.suptitle('Performance Comparison on Synthetic Datasets (V19 Greedy SAS)', fontsize=18)

    try:
        plt.savefig(OUTPUT_PLOT_FILE, format='pdf', dpi=300)
        print(f"Comparison plot saved to: {OUTPUT_PLOT_FILE}")
    except Exception as e:
        print(f"Error saving comparison plot: {e}")

    print("\n--- Synthetic Script finished ---")


if __name__ == "__main__":
    main()

