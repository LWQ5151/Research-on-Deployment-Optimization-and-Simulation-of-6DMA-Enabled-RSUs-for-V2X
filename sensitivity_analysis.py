import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import math
from itertools import combinations  # [V19] 不再需要 combinations
import matplotlib.ticker as mticker

# --- [ 字体配置 ] ---
import matplotlib

try:
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
except Exception as e:
    print(f"Warning: Could not set preferred font. Using default. Error: {e}")
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.titlesize': 16, 'figure.autolayout': True
})
# --- [ 字体配置结束 ] ---

# --- 1. 配置参数 ---
# (只测试我们提出的 DPS-Greedy 方案)
DEPLOYMENT_STRATEGY_NAME = 'dps_greedy'
ANGULAR_HIST_FILE_5MIN = f'results/angular_histograms_{DEPLOYMENT_STRATEGY_NAME}.npy'
DEPLOYMENT_FILE = f'deploy_results/deployment_{DEPLOYMENT_STRATEGY_NAME}.csv'

# (输出目录)
RESULTS_DIR = 'results/sensitivity_analysis/'
RESULTS_CSV_FILE = os.path.join(RESULTS_DIR, 'sensitivity_analysis_results.csv')
RESULTS_PLOT_FILE = os.path.join(RESULTS_DIR, 'sensitivity_analysis_plots.pdf')

# (默认参数 - 用于实验 5)
DEFAULT_M = 3
DEFAULT_W = 60
DEFAULT_TK = 15

# (全局参数)
NUM_ANGLES = 360
EPSILON = 1e-9


# TOP_N_CANDIDATES = 20 # [V19] 不再需要

# --- 2. 核心算法函数 (V19 - 参数化) ---

def find_best_beam(histogram, beam_width_deg):
    """(V19) 查找单个最佳波束的起始角度和分数"""
    padded_hist = np.concatenate((histogram, histogram))
    cum_sum = np.cumsum(padded_hist)
    cum_sum_padded = np.insert(cum_sum, 0, 0)
    scores = cum_sum_padded[beam_width_deg: beam_width_deg + NUM_ANGLES] - \
             cum_sum_padded[0: NUM_ANGLES]

    best_start_angle = np.argmax(scores)
    max_score = scores[best_start_angle]

    return best_start_angle, max_score


def calculate_coverage_score(histogram, angles, beam_width_deg):
    """(V19) 计算给定起始角度的最终覆盖分数"""
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


def run_sas_optimization(histogram, num_beams, beam_width_deg):
    """
    (V19 修正版) 运行 *贪心* SAS 算法
    [修改] 接受 num_beams 和 beam_width_deg 作为参数
    """
    if np.sum(histogram) < EPSILON:
        return [np.nan] * num_beams

    angles = []
    hist_to_optimize = histogram.copy()

    for _ in range(num_beams):
        best_angle, score = find_best_beam(hist_to_optimize, beam_width_deg)
        if score < EPSILON:
            angles.append(np.nan)
        else:
            angles.append(best_angle)
            hist_to_optimize = mask_histogram(hist_to_optimize, best_angle, beam_width_deg)

    while len(angles) < num_beams:
        angles.append(np.nan)

    return angles


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


# --- 4. 核心评估函数 (参数化) ---

def aggregate_histograms(hist_5min, target_timeslot_min):
    """
    将 5 分钟直方图聚合到 target_timeslot_min
    """
    if target_timeslot_min == 5:
        return hist_5min, (24 * 60) // 5  # 288

    agg_factor = target_timeslot_min // 5
    if 288 % agg_factor != 0:
        raise ValueError(f"无法将 288 个 5 分钟时隙均分为 {target_timeslot_min} 分钟的时隙。")

    num_rsus, _, num_angles = hist_5min.shape
    eval_num_timeslots = 288 // agg_factor

    try:
        # 重塑: (num_rsus, new_K, agg_factor, num_angles)
        reshaped = hist_5min.reshape((num_rsus, eval_num_timeslots, agg_factor, num_angles))
        # 沿 agg_factor 轴求和
        aggregated_hist = reshaped.sum(axis=2)
        return aggregated_hist, eval_num_timeslots
    except Exception as e:
        print(f"Error aggregating to {target_timeslot_min} min: {e}")
        return None, 0


def evaluate_scenario(hist_5min, rsus_df, M, W, Tk, return_per_rsu=False):
    """
    在指定参数下运行 A/B/C 评估
    Args:
        hist_5min: 原始 5 分钟直方图 (R, 288, 360)
        rsus_df: 部署文件
        M: 天线数量
        W: 天线宽度
        Tk: 评估时隙粒度 (分钟)
        return_per_rsu: 如果为 True, 返回 RSU 列表；否则返回总和。
    """
    print(f"\n--- Evaluating Scenario: M={M}, W={W}, Tk={Tk} ---")
    num_rsus = len(rsus_df)
    rsu_ids = rsus_df['id'].values

    # 1. 聚合数据到目标时隙
    hist_eval, EVAL_NUM_TS = aggregate_histograms(hist_5min, Tk)
    if hist_eval is None: return None, None, None
    print(f"  Aggregated to {EVAL_NUM_TS} timeslots of {Tk} min.")

    # 2. 计算策略 A (静态) 的角度
    hist_total_day = np.sum(hist_eval, axis=1)  # (R, 360)
    static_angles_all_rsus = [run_sas_optimization(hist_total_day[r], M, W) for r in range(num_rsus)]
    print("  Calculated Static-Optimal (A) angles.")

    # 3. 计算策略 B (预测) 的角度
    #    [设计决策] 策略 B 的预测粒度应与评估粒度 Tk 匹配
    hist_pred = hist_eval  # B(k) 基于 B(k-1)
    dynamic_angles_predictive = np.zeros((num_rsus, EVAL_NUM_TS, M))  # 存储 NaN
    for r in range(num_rsus):
        for k in range(EVAL_NUM_TS):
            k_prev = (k - 1 + EVAL_NUM_TS) % EVAL_NUM_TS
            hist_k_prev = hist_pred[r, k_prev, :]
            dynamic_angles_predictive[r, k, :] = run_sas_optimization(hist_k_prev, M, W)
    print("  Calculated Predictive-Dynamic (B) angles.")

    # 4. 主评估循环
    per_rsu_scores_A = np.zeros(num_rsus)
    per_rsu_scores_B = np.zeros(num_rsus)
    per_rsu_scores_C = np.zeros(num_rsus)

    for r in range(num_rsus):
        static_angles = static_angles_all_rsus[r]
        for k in range(EVAL_NUM_TS):
            hist_k_eval = hist_eval[r, k, :]  # 真实数据

            # 评估 A (静态)
            per_rsu_scores_A[r] += calculate_coverage_score(hist_k_eval, static_angles, W)

            # 评估 B (预测)
            angles_B = dynamic_angles_predictive[r, k, :]
            per_rsu_scores_B[r] += calculate_coverage_score(hist_k_eval, angles_B, W)

            # 评估 C (Oracle)
            angles_C = run_sas_optimization(hist_k_eval, M, W)
            per_rsu_scores_C[r] += calculate_coverage_score(hist_k_eval, angles_C, W)

    print(f"  Evaluation loop complete for M={M}, W={W}, Tk={Tk}.")

    if return_per_rsu:
        return per_rsu_scores_A, per_rsu_scores_B, per_rsu_scores_C
    else:
        return np.sum(per_rsu_scores_A), np.sum(per_rsu_scores_B), np.sum(per_rsu_scores_C)


# --- 5. 主执行函数 ---

def main():
    # 0. 确保输出目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_experiment_results = []  # 存储所有实验结果的列表

    # 1. 加载核心数据
    print("--- Loading Core Data ---")
    try:
        rsus_df = pd.read_csv(DEPLOYMENT_FILE)
        angular_hist_5min = np.load(ANGULAR_HIST_FILE_5MIN)
        print(f"Loaded deployment file: {DEPLOYMENT_FILE} ({len(rsus_df)} RSUs)")
        print(f"Loaded 5-min histogram: {ANGULAR_HIST_FILE_5MIN} (Shape: {angular_hist_5min.shape})")
    except FileNotFoundError as e:
        print(f"Error: Cannot load input file {e.filename}.")
        print("Please ensure deployment file and 5-min angular histogram exist.")
        return

    # 2. [实验 1] 改变天线数量 M
    print("\n--- Experiment 1: Varying Number of Beams (M) ---")
    exp_name = "Varying M (W=60, Tk=5)"
    m_params = [1, 2, 3]
    for m in m_params:
        total_A, total_B, total_C = evaluate_scenario(angular_hist_5min, rsus_df, M=m, W=60, Tk=5)
        all_experiment_results.append(
            {'Experiment': exp_name, 'Parameter': m, 'Score_A_Static': total_A, 'Score_B_Predictive': total_B,
             'Score_C_Oracle': total_C})

    # 3. [实验 2] 改变天线宽度 W (当 M=1 时)
    print("\n--- Experiment 2: Varying Beam Width (W) (M=1) ---")
    exp_name = "Varying W (M=1, Tk=5)"
    w_params = [30, 45, 60, 90, 120, 180]
    for w in w_params:
        total_A, total_B, total_C = evaluate_scenario(angular_hist_5min, rsus_df, M=1, W=w, Tk=5)
        all_experiment_results.append(
            {'Experiment': exp_name, 'Parameter': w, 'Score_A_Static': total_A, 'Score_B_Predictive': total_B,
             'Score_C_Oracle': total_C})

    # 4. [实验 4] 改变时隙粒度 Tk
    print("\n--- Experiment 4: Varying Timeslot Duration (Tk) ---")
    exp_name = "Varying Tk (M=3, W=60)"
    tk_params = [5, 10, 15, 30]
    for tk in tk_params:
        total_A, total_B, total_C = evaluate_scenario(angular_hist_5min, rsus_df, M=DEFAULT_M, W=DEFAULT_W, Tk=tk)
        all_experiment_results.append(
            {'Experiment': exp_name, 'Parameter': tk, 'Score_A_Static': total_A, 'Score_B_Predictive': total_B,
             'Score_C_Oracle': total_C})

    # 5. [实验 5] 改变 RSU 数量 N
    print("\n--- Experiment 5: Varying Number of RSUs (N) ---")
    exp_name = "Varying N (M=3, W=60, Tk=15)"
    n_params = [8, 16, 24, 32]
    # 5a. 先获取默认参数下的 *逐 RSU* 得分
    print(f"  Running default scenario (M={DEFAULT_M}, W={DEFAULT_W}, Tk={DEFAULT_TK}) to get per-RSU scores...")
    scores_A_per_rsu, scores_B_per_rsu, scores_C_per_rsu = evaluate_scenario(
        angular_hist_5min, rsus_df, M=DEFAULT_M, W=DEFAULT_W, Tk=DEFAULT_TK, return_per_rsu=True
    )
    # 5b. 聚合得分
    print("  Aggregating scores for different N...")
    for n in n_params:
        if n > len(rsus_df):
            print(f"  Skipping N={n} (greater than deployed RSUs {len(rsus_df)})")
            continue
        # [关键] 假设 rsus_df (及 per_rsu_scores) 已按贪心部署顺序排列
        total_A = np.sum(scores_A_per_rsu[:n])
        total_B = np.sum(scores_B_per_rsu[:n])
        total_C = np.sum(scores_C_per_rsu[:n])
        all_experiment_results.append(
            {'Experiment': exp_name, 'Parameter': n, 'Score_A_Static': total_A, 'Score_B_Predictive': total_B,
             'Score_C_Oracle': total_C})

    # 6. 保存所有结果到 CSV
    print(f"\n--- Saving all sensitivity results to {RESULTS_CSV_FILE} ---")
    results_df = pd.DataFrame(all_experiment_results)
    # 计算增益百分比
    results_df['Realized_Gain_Pct (B/A-1)'] = (results_df['Score_B_Predictive'] / (
                results_df['Score_A_Static'] + EPSILON) - 1) * 100
    results_df['Potential_Gain_Pct (C/A-1)'] = (results_df['Score_C_Oracle'] / (
                results_df['Score_A_Static'] + EPSILON) - 1) * 100
    try:
        results_df.to_csv(RESULTS_CSV_FILE, index=False, float_format='%.2f')
        print("  Results CSV saved successfully.")
    except Exception as e:
        print(f"  Error saving results CSV: {e}")

    # 7. 可视化所有结果到 PDF
    print(f"--- Generating sensitivity plots to {RESULTS_PLOT_FILE} ---")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 图表
        fig.suptitle(f"Sensitivity Analysis for {DEPLOYMENT_STRATEGY_NAME} Deployment", fontsize=16)

        # 绘图 1: Varying M
        ax1 = axes[0, 0]
        df_m = results_df[results_df['Experiment'] == "Varying M (W=60, Tk=5)"]
        ax1.plot(df_m['Parameter'], df_m['Score_A_Static'], 'bo-', label='A: Static-Optimal')
        ax1.plot(df_m['Parameter'], df_m['Score_B_Predictive'], 'o--', color='orange', label='B: Predictive-Dynamic')
        ax1.plot(df_m['Parameter'], df_m['Score_C_Oracle'], 'g^-', label='C: Oracle-Dynamic')
        ax1.set_xlabel('Number of Beams (M)')
        ax1.set_ylabel('Total GPS Points Covered')
        ax1.set_title('Exp 1: vs. Number of Beams')
        ax1.set_xticks(m_params)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        # 绘图 2: Varying W
        ax2 = axes[0, 1]
        df_w = results_df[results_df['Experiment'] == "Varying W (M=1, Tk=5)"]
        ax2.plot(df_w['Parameter'], df_w['Score_A_Static'], 'bo-', label='A: Static-Optimal')
        ax2.plot(df_w['Parameter'], df_w['Score_B_Predictive'], 'o--', color='orange', label='B: Predictive-Dynamic')
        ax2.plot(df_w['Parameter'], df_w['Score_C_Oracle'], 'g^-', label='C: Oracle-Dynamic')
        ax2.set_xlabel('Beam Width (W) in Degrees (for M=1)')
        ax2.set_ylabel('Total GPS Points Covered')
        ax2.set_title('Exp 2: vs. Beam Width (M=1)')
        ax2.set_xticks(w_params)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        # 绘图 3: Varying Tk
        ax3 = axes[1, 0]
        df_tk = results_df[results_df['Experiment'] == "Varying Tk (M=3, W=60)"]
        ax3.plot(df_tk['Parameter'], df_tk['Score_A_Static'], 'bo-', label='A: Static-Optimal')
        ax3.plot(df_tk['Parameter'], df_tk['Score_B_Predictive'], 'o--', color='orange', label='B: Predictive-Dynamic')
        ax3.plot(df_tk['Parameter'], df_tk['Score_C_Oracle'], 'g^-', label='C: Oracle-Dynamic')
        ax3.set_xlabel('Timeslot Duration (Tk) in Minutes')
        ax3.set_ylabel('Total GPS Points Covered')
        ax3.set_title('Exp 4: vs. Timeslot Duration')
        ax3.set_xticks(tk_params)
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax3.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        # 绘图 4: Varying N
        ax4 = axes[1, 1]
        df_n = results_df[results_df['Experiment'] == "Varying N (M=3, W=60, Tk=15)"]
        ax4.plot(df_n['Parameter'], df_n['Score_A_Static'], 'bo-', label='A: Static-Optimal')
        ax4.plot(df_n['Parameter'], df_n['Score_B_Predictive'], 'o--', color='orange', label='B: Predictive-Dynamic')
        ax4.plot(df_n['Parameter'], df_n['Score_C_Oracle'], 'g^-', label='C: Oracle-Dynamic')
        ax4.set_xlabel('Number of Deployed RSUs (N)')
        ax4.set_ylabel('Total GPS Points Covered')
        ax4.set_title('Exp 5: vs. Number of RSUs')
        ax4.set_xticks(n_params)
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.5)
        ax4.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax4.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 为总标题留出空间
        plt.savefig(RESULTS_PLOT_FILE, format='pdf', dpi=300)
        print(f"  Sensitivity plots saved successfully to {RESULTS_PLOT_FILE}")

    except Exception as e:
        print(f"  Error during plotting: {e}")

    print("\n--- Sensitivity Analysis Script Finished ---")


if __name__ == "__main__":
    main()

