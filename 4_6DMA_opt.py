import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import math
from pyproj import Transformer
from scipy.spatial.distance import cdist  # 用于预处理
import matplotlib.ticker as mticker  # 用于绘图


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
    'xtick.labelsize': 10, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'figure.titlesize': 18
})
# --- [ 字体配置结束 ] ---

# --- 1. 配置参数 ---

# (指定要评估的部署策略名称)
DEPLOYMENT_STRATEGY_NAME = 'dps_greedy'
# DEPLOYMENT_STRATEGY_NAME = 'traffic_greedy'
# DEPLOYMENT_STRATEGY_NAME = 'i_rsu'
# DEPLOYMENT_STRATEGY_NAME = 'random'

# (输入文件)
RAW_GPS_FILE = r"D:\科研相关存档\数据集合集\滴滴成都\GPS_WGS84_20161001.txt"
METADATA_FILE = 'data/grid_metadata.npy'
DEPLOYMENT_FILE = f'deploy_results/deployment_{DEPLOYMENT_STRATEGY_NAME}.csv'

# (输出/缓存文件)
RESULTS_DIR = 'results/'
# [修改] 评估粒度为 5 分钟
INPUT_TIME_WINDOW_MINUTES = 5
EVALUATION_TIME_WINDOW_MINUTES = 5
INPUT_NUM_TIMESLOTS = (24 * 60) // INPUT_TIME_WINDOW_MINUTES  # 288
EVALUATION_NUM_TIMESLOTS = (24 * 60) // EVALUATION_TIME_WINDOW_MINUTES  # 288
# [修改] 不再需要聚合
# AGGREGATION_FACTOR = EVALUATION_TIME_WINDOW_MINUTES // INPUT_TIME_WINDOW_MINUTES

# 预处理缓存 (5分钟)
ANGULAR_HIST_FILE_5MIN = os.path.join(RESULTS_DIR, f'angular_histograms_{DEPLOYMENT_STRATEGY_NAME}.npy')
# 策略C缓存 (5分钟)
DYNAMIC_ANGLES_ORACLE_FILE_5MIN = os.path.join(RESULTS_DIR, f'dynamic_angles_oracle_{DEPLOYMENT_STRATEGY_NAME}.csv')
# 策略B缓存 (5分钟)
DYNAMIC_ANGLES_FILE_5MIN = os.path.join(RESULTS_DIR, f'dynamic_angles_{DEPLOYMENT_STRATEGY_NAME}.csv')
# 策略A缓存 (5分钟聚合)
STATIC_ANGLES_FILE_5MIN = os.path.join(RESULTS_DIR, f'static_angles_{DEPLOYMENT_STRATEGY_NAME}_allday.csv')

# 最终评估输出 (5分钟)
COMPARISON_RESULTS_FILE = os.path.join(RESULTS_DIR, f'evaluation_results_{DEPLOYMENT_STRATEGY_NAME}_5min.csv')
COMPARISON_PLOT_FILE = os.path.join(RESULTS_DIR, f'evaluation_plot_{DEPLOYMENT_STRATEGY_NAME}_5min.pdf')

# (天线参数)
BEAM_WIDTH_DEGREES = 60
NUM_ANGLES = 360
NUM_BEAMS = 3
EPSILON = 1e-9


# (预处理参数)
RSU_COVERAGE_RADIUS_METERS = 500.0
CHUNK_SIZE = 500_000


# --- 2. 核心算法函数 (SAS 及其辅助函数) ---

def find_best_beam(histogram, beam_width_deg):
    """
    (V19 修正版) 使用 np.cumsum 高效找到最佳波束
    返回: (最佳 *起始* 角度, 覆盖的分数)
    """
    # 验证: cumsum 逻辑
    # padded_hist = [H0, H1, ..., H359, H0, H1, ..., H359] (len 720)
    padded_hist = np.concatenate((histogram, histogram))
    # cum_sum = [H0, H0+H1, ..., sum(H0..H359), sum(H0..H359)+H0, ...] (len 720)
    cum_sum = np.cumsum(padded_hist)
    # cum_sum_padded = [0, H0, H0+H1, ...] (len 721)
    cum_sum_padded = np.insert(cum_sum, 0, 0)

    scores = cum_sum_padded[beam_width_deg: beam_width_deg + NUM_ANGLES] - \
             cum_sum_padded[0: NUM_ANGLES]

    # 4. 找到最佳的 *起始* 角度
    best_start_angle = np.argmax(scores)
    max_score = scores[best_start_angle]

    return best_start_angle, max_score


def mask_histogram(histogram, start_angle_in, beam_width_deg):
    """
    (V19 修正版) 安全地掩码一个波束，从 *起始* 角度开始
    """
    hist_copy = histogram.copy()
    if start_angle_in is None or np.isnan(start_angle_in):
        return hist_copy

    start_angle = int(np.round(start_angle_in))

    # [修改] 结束角度 = 起始角度 + 宽度 - 1
    end_angle = (start_angle + beam_width_deg - 1) % NUM_ANGLES

    if start_angle <= end_angle:
        # e.g., start 10 -> [10, 69]
        hist_copy[start_angle: end_angle + 1] = 0  # 切片 [10:70] (即 10..69)
    else:
        # e.g., start 350 -> [350, 359] 和 [0, 48]
        # end_angle = (350 + 60 - 1) % 360 = 409 % 360 = 49
        hist_copy[start_angle: NUM_ANGLES] = 0  # [350, 359]
        hist_copy[0: end_angle + 1] = 0  # [0, 49]
    return hist_copy


def run_sas_optimization(histogram):
    """
    (V19 修正版) 运行SAS算法找到M个波束的 *起始* 角度
    """
    angles = []
    hist_to_optimize = histogram.copy()
    for _ in range(NUM_BEAMS):
        # [修改] 接收 best_start_angle
        best_start_angle, score = find_best_beam(hist_to_optimize, BEAM_WIDTH_DEGREES)
        if score < EPSILON:
            angles.append(np.nan)
        else:
            angles.append(best_start_angle)  # 存储起始角度
            if best_start_angle is not None and not np.isnan(best_start_angle):
                # [修改] 传递起始角度
                hist_to_optimize = mask_histogram(hist_to_optimize, best_start_angle, BEAM_WIDTH_DEGREES)
    while len(angles) < NUM_BEAMS:
        angles.append(np.nan)
    return angles

def calculate_coverage_score(histogram, angles, beam_width_deg):
    """
    (V19 修正版) 计算给定 *起始* 角度的最终覆盖分数
    """
    coverage_mask = np.zeros(NUM_ANGLES, dtype=bool)
    for start_angle_in in angles:  # 'angles' 现在是起始角度列表
        if start_angle_in is None or np.isnan(start_angle_in): continue
        start_angle = int(np.round(start_angle_in))

        # [修改] 结束角度 = 起始角度 + 宽度 - 1
        end_angle = (start_angle + beam_width_deg - 1) % NUM_ANGLES

        if start_angle <= end_angle:
            coverage_mask[start_angle: end_angle + 1] = True
        else:
            coverage_mask[start_angle: NUM_ANGLES] = True
            coverage_mask[0: end_angle + 1] = True
    return np.sum(histogram[coverage_mask])


# --- 3. 预处理函数 (从 V14 复制) ---
# ... (get_or_create_angular_histograms 函数保持不变) ...
def get_or_create_angular_histograms(rsus_df, metadata, raw_gps_file, output_hist_file):
    """
    加载或创建 5 分钟粒度的精确角度直方图
    """
    print(f"\nChecking for 5-min angular histogram file: {output_hist_file}")
    if os.path.exists(output_hist_file):
        print("  File exists. Loading...")
        try:
            angular_histograms = np.load(output_hist_file)
            if angular_histograms.shape == (len(rsus_df), INPUT_NUM_TIMESLOTS, NUM_ANGLES):
                print("  Cached file loaded successfully (shape matches).")
                return angular_histograms
            else:
                print(f"  Warning: Cached histogram shape mismatch. Recalculating...")
        except Exception as e:
            print(f"  Error loading cached histogram: {e}. Recalculating...")

    print("  File not found or invalid. Running preprocessing...")
    # --- (V14 的 preprocess_data_for_optimization_accurate 函数) ---
    print("--- Running Accurate Preprocessing (from raw GPS) ---")
    overall_start_time = time.time()
    num_rsus = len(rsus_df)
    crs_wgs84 = metadata.get('crs_wgs84', "EPSG:4326")
    crs_utm = metadata.get('crs_utm', "EPSG:32648")
    transformer_wgs_to_utm = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)
    angular_histograms = np.zeros((num_rsus, INPUT_NUM_TIMESLOTS, NUM_ANGLES), dtype=np.uint32)
    rsu_utm_coords = rsus_df[['utm_x', 'utm_y']].values
    total_points_processed = 0
    file_processed_successfully = False
    try:
        iterator = pd.read_csv(
            raw_gps_file, header=None, names=['vehicle_id', 'order_id', 'timestamp', 'longitude', 'latitude'],
            chunksize=CHUNK_SIZE, low_memory=False, on_bad_lines='warn'
        )
        for chunk_df in iterator:
            chunk_start_time = time.time()
            print(
                f"  Processing GPS chunk (Rows {total_points_processed + 1} - {total_points_processed + len(chunk_df)})...")
            try:
                chunk_df['timestamp'] = pd.to_numeric(chunk_df['timestamp'], errors='coerce')
                chunk_df = chunk_df.dropna(subset=['timestamp'])
                datetimes = pd.to_datetime(chunk_df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Shanghai')
                chunk_df['timeslot_k'] = (datetimes.dt.hour * (60 // INPUT_TIME_WINDOW_MINUTES)) + \
                                         (datetimes.dt.minute // INPUT_TIME_WINDOW_MINUTES)
                chunk_df = chunk_df[(chunk_df['timeslot_k'] >= 0) & (chunk_df['timeslot_k'] < INPUT_NUM_TIMESLOTS)]
            except Exception as e:
                print(f"    Warning: Error processing timestamps: {e}. Skipping chunk.");
                total_points_processed += CHUNK_SIZE;
                continue
            try:
                chunk_df['longitude'] = pd.to_numeric(chunk_df['longitude'], errors='coerce')
                chunk_df['latitude'] = pd.to_numeric(chunk_df['latitude'], errors='coerce')
                chunk_df = chunk_df.dropna(subset=['longitude', 'latitude'])
                utm_x, utm_y = transformer_wgs_to_utm.transform(chunk_df['longitude'].values,
                                                                chunk_df['latitude'].values)
                chunk_df['utm_x'] = utm_x;
                chunk_df['utm_y'] = utm_y
            except Exception as e:
                print(f"    Warning: Error converting coordinates: {e}. Skipping chunk.");
                total_points_processed += len(chunk_df);
                continue
            if chunk_df.empty:
                print("    Chunk empty after cleaning. Skipping.");
                total_points_processed += CHUNK_SIZE;
                continue
            points_utm = chunk_df[['utm_x', 'utm_y']].values
            point_timeslots = chunk_df['timeslot_k'].values
            dist_sq_matrix = cdist(points_utm, rsu_utm_coords, metric='sqeuclidean')
            in_range_mask = dist_sq_matrix <= (RSU_COVERAGE_RADIUS_METERS ** 2)
            point_indices, rsu_indices = np.nonzero(in_range_mask)
            if len(point_indices) > 0:
                points_covered_utm = points_utm[point_indices];
                k_covered = point_timeslots[point_indices]
                rsu_covered_coords = rsu_utm_coords[rsu_indices]
                dx = points_covered_utm[:, 0] - rsu_covered_coords[:, 0];
                dy = points_covered_utm[:, 1] - rsu_covered_coords[:, 1]
                angles_rad = np.arctan2(dy, dx);
                angles_deg = np.degrees(angles_rad)
                angles_norm = (angles_deg + 360) % 360;
                angle_bins = np.round(angles_norm).astype(int) % NUM_ANGLES
                unique_rka, counts = np.unique(np.stack((rsu_indices, k_covered, angle_bins), axis=-1), axis=0,
                                               return_counts=True)
                r_idx = unique_rka[:, 0].astype(int);
                k_idx = unique_rka[:, 1].astype(int);
                a_idx = unique_rka[:, 2].astype(int)
                np.add.at(angular_histograms, (r_idx, k_idx, a_idx), counts)
            total_points_processed += len(chunk_df)
            print(
                f"    ...Chunk processed in {time.time() - chunk_start_time:.2f}s. Total processed: {total_points_processed}")
            file_processed_successfully = True
    except FileNotFoundError:
        print(f"Error: Raw GPS file not found at {raw_gps_file}"); return None
    except pd.errors.EmptyDataError:
        print(f"Error: Raw GPS file {raw_gps_file} is empty or badly formatted."); return None
    except Exception as e:
        print(f"Unexpected error during GPS file processing: {e}");
    if not file_processed_successfully: print("Error: No GPS data processed."); return None
    print(f"\n--- Accurate preprocessing finished. Total time: {time.time() - overall_start_time:.2f}s ---")
    try:
        os.makedirs(os.path.dirname(output_hist_file), exist_ok=True)
        np.save(output_hist_file, angular_histograms)
        print(f"Accurate angular histograms saved to: {output_hist_file}")
    except Exception as e:
        print(f"Error saving histogram file {output_hist_file}: {e}"); return None
    return angular_histograms
    # --- (V14 预处理函数结束) ---


# --- 4. 策略计算函数 (修改为 5 分钟) ---
# ... (get_or_create_static_angles_5min_allday 函数保持不变) ...
def get_or_create_static_angles_5min_allday(angular_histograms_5min, rsus_df, output_csv_file):
    """[修改] 计算或加载 5 分钟聚合的静态角度 (全天)"""
    print(f"\nCalculating/Loading Static-Optimal angles (5-min All-Day)...")
    if os.path.exists(output_csv_file):
        print(f"  Loading existing file: {output_csv_file}")
        try:
            static_angles_df = pd.read_csv(output_csv_file).set_index('rsu_id')
            if len(static_angles_df) == len(rsus_df):
                print("  Static angles file loaded successfully.")
                return static_angles_df
            else:
                raise ValueError(f"RSU count mismatch")
        except Exception as e:
            print(f"  Error loading static angles file or file invalid: {e}. Recalculating...")

    print(f"  Static angles file not found or invalid. Calculating...")
    start_static_time = time.time()
    static_angles_results = []
    rsu_ids = rsus_df['id'].values
    for r in range(len(rsus_df)):
        rsu_id = rsu_ids[r]
        # [修改] 对 5 分钟直方图的 288 个时隙求和
        hist_total_day_5min = np.sum(angular_histograms_5min[r, :, :], axis=0)
        angles = run_sas_optimization(hist_total_day_5min)
        angles_for_csv = [a if a is not None else np.nan for a in angles]
        static_angles_results.append({
            'rsu_id': rsu_id, 'grid_i': rsus_df.iloc[r]['grid_i'], 'grid_j': rsus_df.iloc[r]['grid_j'],
            'static_start_angle_1': angles_for_csv[0],  # [修改] 重命名列
            'static_start_angle_2': angles_for_csv[1],
            'static_start_angle_3': angles_for_csv[2]
        })
    static_angles_df = pd.DataFrame(static_angles_results).set_index('rsu_id')
    try:
        static_angles_df.to_csv(output_csv_file, float_format='%.1f')
        print(f"  New Static angles saved to '{output_csv_file}'. Time: {time.time() - start_static_time:.2f}s")
    except Exception as e:
        print(f"  Error saving static angles file: {e}")
    return static_angles_df


# ... (get_or_create_dynamic_angles_5min 函数保持不变) ...
def get_or_create_dynamic_angles_5min(angular_histograms_5min, rsus_df, output_csv_file):
    """计算或加载 5 分钟动态角度"""
    print(f"\nCalculating/Loading Predictive-Dynamic angles (5-min)...")
    if os.path.exists(output_csv_file):
        print(f"  Loading existing file: {output_csv_file}")
        try:
            dynamic_angles_df = pd.read_csv(output_csv_file).set_index(['rsu_id', 'timeslot_k'])
            expected_len = len(rsus_df) * INPUT_NUM_TIMESLOTS
            if len(dynamic_angles_df) == expected_len:
                print("  5-min dynamic angles file loaded successfully.")
                return dynamic_angles_df
            else:
                raise ValueError(
                    f"Dynamic angles file size mismatch (found {len(dynamic_angles_df)}, expected {expected_len})")
        except Exception as e:
            print(f"  Error loading dynamic angles file: {e}. Recalculating...")

    print(f"  5-min dynamic angles file not found or invalid. Calculating...")
    start_time = time.time()
    num_rsus, num_timeslots_5min, _ = angular_histograms_5min.shape
    dynamic_angles_results = []
    rsu_ids = rsus_df['id'].values
    for r in range(num_rsus):
        rsu_id = rsu_ids[r]
        for k in range(num_timeslots_5min):
            k_data = (k - 1 + num_timeslots_5min) % num_timeslots_5min
            hist_k_prev = angular_histograms_5min[r, k_data, :]
            angles = run_sas_optimization(hist_k_prev)
            angles_for_csv = [a if a is not None else np.nan for a in angles]
            dynamic_angles_results.append({
                'rsu_id': rsu_id, 'timeslot_k': k, 'based_on_k-1': k_data,
                'dynamic_start_angle_1': angles_for_csv[0],  # [修改] 重命名列
                'dynamic_start_angle_2': angles_for_csv[1],
                'dynamic_start_angle_3': angles_for_csv[2]
            })
    df = pd.DataFrame(dynamic_angles_results)
    try:
        df.to_csv(output_csv_file, index=False, float_format='%.1f')
        print(f"--- 5-min dynamic angles calculation complete. Time: {time.time() - start_time:.2f}s ---")
        print(f"5-min dynamic angles saved to: {output_csv_file}")
        return df.set_index(['rsu_id', 'timeslot_k'])
    except Exception as e:
        print(f"Error saving 5-min dynamic angles CSV: {e}")
        return None

def get_or_create_dynamic_angles_oracle_5min(angular_histograms_5min, rsus_df, output_csv_file):
    """计算或加载 5 分钟动态角度"""
    print(f"\nCalculating/Loading Predictive-Dynamic angles oracle (5-min)...")
    if os.path.exists(output_csv_file):
        print(f"  Loading existing file: {output_csv_file}")
        try:
            dynamic_angles_df = pd.read_csv(output_csv_file).set_index(['rsu_id', 'timeslot_k'])
            expected_len = len(rsus_df) * INPUT_NUM_TIMESLOTS
            if len(dynamic_angles_df) == expected_len:
                print("  5-min dynamic angles file loaded successfully.")
                return dynamic_angles_df
            else:
                raise ValueError(
                    f"Dynamic angles file size mismatch (found {len(dynamic_angles_df)}, expected {expected_len})")
        except Exception as e:
            print(f"  Error loading dynamic angles file: {e}. Recalculating...")

    print(f"  5-min dynamic angles oracle file not found or invalid. Calculating...")
    start_time = time.time()
    num_rsus, num_timeslots_5min, _ = angular_histograms_5min.shape
    dynamic_angles_results = []
    rsu_ids = rsus_df['id'].values
    for r in range(num_rsus):
        rsu_id = rsu_ids[r]
        for k in range(num_timeslots_5min):
            #k_data = (k + num_timeslots_5min) % num_timeslots_5min
            hist_k_prev = angular_histograms_5min[r, k, :]
            angles = run_sas_optimization(hist_k_prev)
            angles_for_csv = [a if a is not None else np.nan for a in angles]
            dynamic_angles_results.append({
                'rsu_id': rsu_id, 'timeslot_k': k, 'based_on_k': k,
                'dynamic_start_angle_1': angles_for_csv[0],  # [修改] 重命名列
                'dynamic_start_angle_2': angles_for_csv[1],
                'dynamic_start_angle_3': angles_for_csv[2]
            })
    df = pd.DataFrame(dynamic_angles_results)
    try:
        df.to_csv(output_csv_file, index=False, float_format='%.1f')
        print(f"--- 5-min dynamic angles oracle calculation complete. Time: {time.time() - start_time:.2f}s ---")
        print(f"5-min dynamic angles saved to: {output_csv_file}")
        return df.set_index(['rsu_id', 'timeslot_k'])
    except Exception as e:
        print(f"Error saving 5-min dynamic angles oracle CSV: {e}")
        return None


# --- 5. 主评估流程 (修改为 5 分钟) ---

def main():
    # 0. 确保输出目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. 加载部署文件和元数据
    print("Loading deployment and metadata files...")
    try:
        rsus_df = pd.read_csv(DEPLOYMENT_FILE)
        metadata = np.load(METADATA_FILE, allow_pickle=True).item()
        if rsus_df.empty: raise ValueError(f"Deployment file {DEPLOYMENT_FILE} is empty.")
        if 'covered_potential' not in rsus_df.columns: rsus_df['covered_potential'] = np.nan
    except FileNotFoundError as e:
        print(f"错误: 找不到输入文件 {e.filename}。"); return
    except ValueError as e:
        print(f"加载或验证文件时出错: {e}"); return
    except Exception as e:
        print(f"加载文件时发生未知错误: {e}"); return

    num_rsus = len(rsus_df)
    rsu_ids = rsus_df['id'].values
    print(f"Successfully loaded {num_rsus} RSUs for '{DEPLOYMENT_STRATEGY_NAME}'.")

    # 2. 获取 5 分钟角度直方图 (计算或加载)
    angular_histograms_5min = get_or_create_angular_histograms(rsus_df, metadata, RAW_GPS_FILE, ANGULAR_HIST_FILE_5MIN)
    if angular_histograms_5min is None: print("Error: Failed to obtain 5-min angular histograms. Aborting."); return

    # 3. [移除] 不再需要 15 分钟聚合
    print("\nEvaluation will be performed at 5-minute intervals.")
    # angular_histograms_15min = ...

    # 4. 获取策略 A (静态, 5分钟全天聚合)
    static_angles_df = get_or_create_static_angles_5min_allday(angular_histograms_5min, rsus_df,
                                                               STATIC_ANGLES_FILE_5MIN)

    # 5. 获取策略 B (预测, 5分钟)
    dynamic_angles_df_5min = get_or_create_dynamic_angles_5min(angular_histograms_5min, rsus_df,
                                                               DYNAMIC_ANGLES_FILE_5MIN)

    # 5. 获取策略 C (5分钟)
    dynamic_angles_oracle_df_5min = get_or_create_dynamic_angles_oracle_5min(angular_histograms_5min, rsus_df,
                                                               DYNAMIC_ANGLES_ORACLE_FILE_5MIN)

    if dynamic_angles_df_5min is None: print(
        "Error: Failed to obtain 5-min dynamic angles. Aborting evaluation."); return

    # 6. 模拟和评估 (A vs B vs C @ 5分钟)
    print(f"\n--- Running evaluation for '{DEPLOYMENT_STRATEGY_NAME}' deployment (5-min intervals) ---")
    start_eval_time = time.time()
    results = []
    missing_b_angles_count = 0

    for r in range(num_rsus):
        rsu_id = rsu_ids[r]
        rsu_row = rsus_df.iloc[r]
        rsu_hist_index = r  # 假设索引 'r' 与直方图索引 'r' 对应

        if (r + 1) % 5 == 0 or r == 0: print(f"  Evaluating RSU ID: {rsu_id} (Index: {rsu_hist_index})...")

        try:
            # [修改] 使用新的列名
            static_angles = static_angles_df.loc[rsu_id][
                ['static_start_angle_1', 'static_start_angle_2', 'static_start_angle_3']].values
        except KeyError:
            print(f"    Warning: Cannot find RSU ID {rsu_id} in static angles. Skipping RSU.");
            continue

        total_score_A = 0;
        total_score_B = 0;
        total_score_C = 0

        # [修改] 遍历 288 个 5 分钟时隙
        for k_5min in range(INPUT_NUM_TIMESLOTS):
            # i. 获取 5 分钟的 "真实" 数据
            hist_k_5min = angular_histograms_5min[rsu_hist_index, k_5min, :]

            # --- 评估 A: Static ---
            # (使用基于 5 分钟全天聚合计算的静态角度)
            total_score_A += calculate_coverage_score(hist_k_5min, static_angles, BEAM_WIDTH_DEGREES)

            # --- 评估 B: Predictive ---
            # (查找为当前 5 分钟时隙 k_5min 准备的角度)
            try:
                # [修改] 使用新的列名
                dynamic_angles_predictive = dynamic_angles_df_5min.loc[(rsu_id, k_5min)][
                    ['dynamic_start_angle_1', 'dynamic_start_angle_2', 'dynamic_start_angle_3']].values
                total_score_B += calculate_coverage_score(hist_k_5min, dynamic_angles_predictive, BEAM_WIDTH_DEGREES)
            except KeyError:
                missing_b_angles_count += 1;
                pass

            # --- 评估 B: Predictive ---
            # (查找为当前 5 分钟时隙 k_5min 准备的角度)
            try:
                # [修改] 使用新的列名
                dynamic_angles_oracle_predictive = dynamic_angles_oracle_df_5min.loc[(rsu_id, k_5min)][
                    ['dynamic_start_angle_1', 'dynamic_start_angle_2', 'dynamic_start_angle_3']].values
                total_score_C += calculate_coverage_score(hist_k_5min, dynamic_angles_oracle_predictive,
                                                          BEAM_WIDTH_DEGREES)
            except KeyError:
                missing_b_angles_oracle_count += 1;
                pass

            #
            #     # --- 评估 C: Oracle ---
            # # (直接在 5 分钟数据上计算最优角度)
            # dynamic_angles_oracle = run_sas_optimization(hist_k_5min)
            # total_score_C += calculate_coverage_score(hist_k_5min, dynamic_angles_oracle, BEAM_WIDTH_DEGREES)

        # c. 计算衍生指标
        potential_gain_abs = total_score_C - total_score_A
        potential_gain_pct = (potential_gain_abs / (total_score_A + EPSILON)) * 100 if abs(
            total_score_A) > EPSILON else np.inf if potential_gain_abs > 0 else 0
        realized_gain_abs = total_score_B - total_score_A
        realized_gain_pct = (realized_gain_abs / (total_score_A + EPSILON)) * 100 if abs(
            total_score_A) > EPSILON else np.inf if realized_gain_abs > 0 else 0
        prediction_loss_abs = total_score_C - total_score_B
        prediction_loss_pct = (prediction_loss_abs / (total_score_C + EPSILON)) * 100 if abs(
            total_score_C) > EPSILON else 0
        covered_potential = rsu_row.get('covered_potential', np.nan)

        results.append({
            'DeploymentStrategy': DEPLOYMENT_STRATEGY_NAME, 'RSU_ID': rsu_id,
            'Grid_i': rsu_row.get('grid_i', np.nan), 'Grid_j': rsu_row.get('grid_j', np.nan),
            'Score_A_Static_5min': total_score_A, 'Score_B_Predictive_5min': total_score_B,
            'Score_C_Oracle_5min': total_score_C,
            'PotentialGain_Abs (C-A)': potential_gain_abs, 'PotentialGain_Pct ((C-A)/A)': potential_gain_pct,
            'RealizedGain_Abs (B-A)': realized_gain_abs, 'RealizedGain_Pct ((B-A)/A)': realized_gain_pct,
            'PredictionLoss_Abs (C-B)': prediction_loss_abs, 'PredictionLoss_Pct ((C-B)/C)': prediction_loss_pct,
            'CoveredPotential_at_Deployment': covered_potential
        })

    print(f"--- Evaluation completed, total time: {time.time() - start_eval_time:.2f} seconds ---")
    if missing_b_angles_count > 0: print(
        f"    Warning: Encountered {missing_b_angles_count} instances where predictive angles (Strategy B) were missing.")

    # 7. 保存结果到 CSV
    if results:
        results_df = pd.DataFrame(results).set_index('RSU_ID')
        try:
            results_df.to_csv(COMPARISON_RESULTS_FILE, float_format='%.2f'); print(
                f"\nComprehensive 5-min evaluation results saved to: {COMPARISON_RESULTS_FILE}")
        except Exception as e:
            print(f"\nError saving results CSV: {e}")
    else:
        print("\nNo results generated."); return

    # 8. 打印全局总结
    total_A = results_df['Score_A_Static_5min'].sum();
    total_B = results_df['Score_B_Predictive_5min'].sum();
    total_C = results_df['Score_C_Oracle_5min'].sum()
    print("\n--- Global Coverage Performance Summary (5-min Evaluation) ---")
    print(f"  Deployment Strategy: {DEPLOYMENT_STRATEGY_NAME}")
    print(f"  [A] Static-Optimal (5min): {total_A:,.0f} GPS Points");
    print(f"  [B] Predictive-Dynamic (5min): {total_B:,.0f} GPS Points");
    print(f"  [C] Oracle-Dynamic (5min): {total_C:,.0f} GPS Points");
    print(f"  ---------------------------------")
    if abs(total_A) > EPSILON:
        potential_gain_pct_total = ((total_C / total_A) - 1) * 100; realized_gain_pct_total = ((
                                                                                                           total_B / total_A) - 1) * 100
    else:
        potential_gain_pct_total = np.inf if total_C > total_A else 0; realized_gain_pct_total = np.inf if total_B > total_A else 0
    print(f"  Max Potential Gain (C vs A): {potential_gain_pct_total:+.2f} %");
    print(f"  Realized Gain (B vs A): {realized_gain_pct_total:+.2f} %");
    print("----------------------------\n")

    # 9. 可视化
    print(f"Generating comparison plot for '{DEPLOYMENT_STRATEGY_NAME}' deployment (5-min evaluation)...")
    plot_df = results_df.sort_index();
    n_groups = len(plot_df);
    index = np.arange(n_groups);
    bar_width = 0.25
    plt.figure(figsize=(max(15, n_groups * 0.7), 8))
    rects1 = plt.bar(index - bar_width, plot_df['Score_A_Static_5min'], bar_width, label='A: Static-Optimal (5min)',
                     color='b', alpha=0.7)
    rects2 = plt.bar(index, plot_df['Score_B_Predictive_5min'], bar_width, label='B: Predictive-Dynamic (5min)',
                     color='orange', alpha=0.7)
    rects3 = plt.bar(index + bar_width, plot_df['Score_C_Oracle_5min'], bar_width, label='C: Oracle-Dynamic (5min)',
                     color='g', alpha=0.7)
    plt.xlabel('RSU ID');
    plt.ylabel('Total GPS Points Covered (24h @ 5min interval)');
    plt.title(f'Coverage Performance Comparison (5min Eval) for {DEPLOYMENT_STRATEGY_NAME} Deployment')
    plt.xticks(index, plot_df.index.astype(int), rotation=45, ha='right');
    plt.legend();
    plt.grid(axis='y', linestyle='--', alpha=0.5);
    plt.tight_layout()
    try:
        plt.savefig(COMPARISON_PLOT_FILE, format='pdf', dpi=300); print(
            f"Comparison plot saved to: {COMPARISON_PLOT_FILE}")
    except Exception as e:
        print(f"Error saving comparison plot: {e}")
    plt.show()
    print("\n--- Script finished ---")


if __name__ == "__main__":
    main()

