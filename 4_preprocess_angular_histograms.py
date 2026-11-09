import numpy as np
import pandas as pd
from pyproj import Transformer
import os
import time
import math
from scipy.spatial.distance import cdist # 用于高效计算距离

# --- 1. 配置参数 ---

# (输入文件)
RAW_GPS_FILE = r"D:\科研相关存档\数据集合集\滴滴成都\GPS_WGS84_20161001.txt"
METADATA_FILE = 'data/grid_metadata.npy'
DEPLOYMENT_DIR = 'deploy_results/'
# [修改] 包含我们自己的方案，并指定输出文件名
DEPLOYMENT_FILES = {
    'dps_greedy': os.path.join(DEPLOYMENT_DIR, 'deployment_dps_greedy.csv'),
    'traffic_greedy': os.path.join(DEPLOYMENT_DIR, 'deployment_traffic_greedy.csv'),
    'i_rsu': os.path.join(DEPLOYMENT_DIR, 'deployment_i_rsu.csv'),
    'random': os.path.join(DEPLOYMENT_DIR, 'deployment_random.csv')
}

# (输出目录和文件名前缀)
OUTPUT_DIR = 'results/'
HIST_FILENAME_PREFIX = 'angular_histograms_' # e.g., angular_histograms_dps_greedy.npy

# (参数)
RSU_COVERAGE_RADIUS_METERS = 500.0
NUM_TIMESLOTS = 288
TIME_WINDOW_MINUTES = 5 #以5min为间隔，方便后续直接合成10,15...
NUM_ANGLES = 360
CHUNK_SIZE = 500_000 # 每次处理 50 万行 GPS 数据

# --- 2. 核心预处理函数 ---

def calculate_angular_histograms(deployment_strategies, metadata, raw_gps_file):
    """
    从原始 GPS 文件为多个部署策略计算精确的角度直方图。

    Args:
        deployment_strategies (dict): key=策略名称, value=包含 RSU 位置的 DataFrame。
        metadata (dict): 网格元数据。
        raw_gps_file (str): 原始 GPS 文件路径。

    Returns:
        dict: key=策略名称, value=计算得到的 3D 角度直方图 NumPy 数组。
              如果某个策略的文件处理失败，则对应的值为 None。
    """
    print("--- 开始精确预处理 (多方案, 从原始 GPS) ---")
    print(f"--- 使用 RSU 半径: {RSU_COVERAGE_RADIUS_METERS}m ---")
    print("--- 注意：此过程可能非常耗时！ ---")
    overall_start_time = time.time()

    # 1. 准备坐标转换器和 RSU UTM 坐标
    crs_wgs84 = metadata.get('crs_wgs84', "EPSG:4326")
    crs_utm = metadata.get('crs_utm', "EPSG:32648")
    transformer_wgs_to_utm = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)

    rsu_coords_utm = {}
    histograms = {}
    rsu_counts = {}
    for name, df in deployment_strategies.items():
        if df is None or df.empty:
            print(f"  策略 '{name}' 数据为空或加载失败，跳过。")
            continue
        # 检查必要的列
        if not all(col in df.columns for col in ['id', 'utm_x', 'utm_y']):
            print(f"  错误: 策略 '{name}' 的 DataFrame 缺少 'id', 'utm_x', 或 'utm_y' 列。跳过。")
            continue
        rsu_coords_utm[name] = df[['utm_x', 'utm_y']].values
        num_rsus = len(rsu_coords_utm[name])
        rsu_counts[name] = num_rsus
        histograms[name] = np.zeros((num_rsus, NUM_TIMESLOTS, NUM_ANGLES), dtype=np.uint32)
        print(f"  为策略 '{name}' 初始化直方图，RSU 数量: {num_rsus}")

    if not histograms:
        print("错误：没有有效的部署策略可供处理。")
        return {}

    # 2. 逐块处理原始 GPS 文件
    total_points_processed = 0
    file_processed_successfully = False

    try:
        iterator = pd.read_csv(
            raw_gps_file,
            header=None,
            names=['vehicle_id', 'order_id', 'timestamp', 'longitude', 'latitude'],
            chunksize=CHUNK_SIZE,
            low_memory=False,
            on_bad_lines='warn' # 记录但不中断坏行
        )

        for chunk_df in iterator:
            chunk_start_time = time.time()
            print(f"\n  处理 GPS 数据块 (行 {total_points_processed + 1} - {total_points_processed + len(chunk_df)})...")

            # a. 清理和转换时间戳 -> k
            try:
                chunk_df['timestamp'] = pd.to_numeric(chunk_df['timestamp'], errors='coerce')
                chunk_df = chunk_df.dropna(subset=['timestamp'])
                datetimes = pd.to_datetime(chunk_df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Shanghai')
                chunk_df['timeslot_k'] = (datetimes.dt.hour * (60 // TIME_WINDOW_MINUTES)) + \
                                         (datetimes.dt.minute // TIME_WINDOW_MINUTES)
                chunk_df = chunk_df[(chunk_df['timeslot_k'] >= 0) & (chunk_df['timeslot_k'] < NUM_TIMESLOTS)]
            except Exception as e:
                print(f"    警告：处理时间戳时出错: {e}。跳过此块。")
                total_points_processed += CHUNK_SIZE # 近似增加计数器
                continue

            # b. 清理和转换坐标 WGS84 -> UTM
            try:
                chunk_df['longitude'] = pd.to_numeric(chunk_df['longitude'], errors='coerce')
                chunk_df['latitude'] = pd.to_numeric(chunk_df['latitude'], errors='coerce')
                chunk_df = chunk_df.dropna(subset=['longitude', 'latitude'])
                utm_x, utm_y = transformer_wgs_to_utm.transform(chunk_df['longitude'].values, chunk_df['latitude'].values)
                chunk_df['utm_x'] = utm_x
                chunk_df['utm_y'] = utm_y
            except Exception as e:
                print(f"    警告：坐标转换时出错: {e}。跳过此块。")
                total_points_processed += len(chunk_df) # 增加实际处理的行数
                continue

            if chunk_df.empty:
                print("    数据块在清理后为空，跳过。")
                total_points_processed += CHUNK_SIZE # 近似
                continue

            points_utm = chunk_df[['utm_x', 'utm_y']].values # Shape: (num_points, 2)
            point_timeslots = chunk_df['timeslot_k'].values

            # c. 为每个策略处理点
            for name, rsu_coords in rsu_coords_utm.items():
                num_rsus = rsu_counts[name]
                hist = histograms[name] # 获取对应策略的直方图数组引用

                # i. 计算所有点到该策略下所有 RSU 的距离平方 (高效)
                # dist_sq_matrix[p, r] = 点 p 到 RSU r 的距离平方
                dist_sq_matrix = cdist(points_utm, rsu_coords, metric='sqeuclidean')

                # ii. 找到哪些点落在了哪个 RSU 的覆盖范围内
                #    in_range_mask[p, r] = True if 点 p 在 RSU r 范围内
                in_range_mask = dist_sq_matrix <= (RSU_COVERAGE_RADIUS_METERS**2)

                # iii. 获取落在范围内的点索引 (point_idx) 和对应的 RSU 索引 (rsu_idx)
                point_indices, rsu_indices = np.nonzero(in_range_mask)

                if len(point_indices) > 0: # 如果这个块中有任何点被覆盖
                    # 提取这些点的 UTM 坐标、时间片和对应的 RSU 坐标
                    points_covered_utm = points_utm[point_indices]
                    k_covered = point_timeslots[point_indices]
                    rsu_covered_coords = rsu_coords[rsu_indices] # (n_covered, 2)

                    # iv. 计算精确角度
                    dx = points_covered_utm[:, 0] - rsu_covered_coords[:, 0]
                    dy = points_covered_utm[:, 1] - rsu_covered_coords[:, 1]
                    angles_rad = np.arctan2(dy, dx)
                    angles_deg = np.degrees(angles_rad)
                    angles_norm = (angles_deg + 360) % 360
                    angle_bins = np.round(angles_norm).astype(int) % NUM_ANGLES

                    # v. 聚合到直方图
                    #    我们有 rsu_indices, k_covered, angle_bins 三个数组 (长度相同)
                    #    需要将它们对应的计数加到 hist[rsu_index, k, angle_bin]
                    unique_rka, counts = np.unique(
                        np.stack((rsu_indices, k_covered, angle_bins), axis=-1),
                        axis=0, return_counts=True
                    )
                    r_idx = unique_rka[:, 0]
                    k_idx = unique_rka[:, 1]
                    a_idx = unique_rka[:, 2]

                    np.add.at(hist, (r_idx, k_idx, a_idx), counts)

            total_points_processed += len(chunk_df)
            print(f"    ...块处理完成，耗时: {time.time() - chunk_start_time:.2f} 秒。累计处理: {total_points_processed} 点。")
            file_processed_successfully = True # 标记至少成功处理了一个块

    except FileNotFoundError:
        print(f"错误: 找不到原始 GPS 文件 {raw_gps_file}")
        return {}
    except pd.errors.EmptyDataError:
        print(f"错误: 原始 GPS 文件 {raw_gps_file} 为空或格式错误。")
        return {}
    except Exception as e:
        print(f"处理 GPS 文件时发生意外错误: {e}")
        # 即使出错也尝试返回已计算的部分
        if not file_processed_successfully:
             return {} # 如果一个块都没处理成功，返回空

    print(f"\n--- 精确预处理完成，总耗时: {time.time() - overall_start_time:.2f} 秒 ---")

    # 3. 保存每个策略的直方图
    final_histograms = {}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, hist_array in histograms.items():
        if hist_array is not None: # 确保数组有效
            output_filename = os.path.join(OUTPUT_DIR, f"{HIST_FILENAME_PREFIX}{name}.npy")
            try:
                np.save(output_filename, hist_array)
                print(f"直方图已保存: {output_filename}")
                final_histograms[name] = hist_array
            except Exception as e:
                print(f"保存直方图 '{name}' 时出错: {e}")
                final_histograms[name] = None # 标记失败
        else:
            final_histograms[name] = None


    return final_histograms

# --- 4. 主执行函数 ---

def main():
    # 1. 加载元数据
    print("Loading metadata...")
    try:
        metadata = np.load(METADATA_FILE, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return

    # 2. 加载所有部署策略文件
    deployment_strategies = {}
    print("\nLoading deployment strategy files...")
    all_files_found = True
    for name, filepath in DEPLOYMENT_FILES.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                # 基本检查
                if df.empty or 'id' not in df.columns or 'utm_x' not in df.columns or 'utm_y' not in df.columns:
                    print(f"  Warning: File '{filepath}' for strategy '{name}' is empty or missing required columns. Skipping.")
                    deployment_strategies[name] = None
                else:
                     deployment_strategies[name] = df
                     print(f"  Loaded '{name}' ({len(df)} RSUs)")
            except Exception as e:
                print(f"  Error loading file '{filepath}' for strategy '{name}': {e}")
                deployment_strategies[name] = None
                all_files_found = False # 标记有文件加载失败
        else:
            print(f"  Error: Deployment file not found for '{name}' at '{filepath}'.")
            deployment_strategies[name] = None
            all_files_found = False

    if not any(df is not None for df in deployment_strategies.values()):
        print("\nError: No valid deployment files loaded. Cannot proceed with preprocessing.")
        return
    elif not all_files_found:
        print("\nWarning: Some deployment files were missing or invalid. Proceeding with available ones.")

    # 3. 执行精确预处理（如果需要）
    #    检查是否所有策略的直方图文件都已存在且有效
    all_hist_exist = True
    hist_files_to_check = {}
    for name in deployment_strategies:
        if deployment_strategies[name] is not None: # 只检查有效策略
             hist_filename = os.path.join(OUTPUT_DIR, f"{HIST_FILENAME_PREFIX}{name}.npy")
             hist_files_to_check[name] = hist_filename
             if not os.path.exists(hist_filename):
                 all_hist_exist = False
                 break # 只要有一个不存在就需要重新计算

    if all_hist_exist:
        print(f"\nAll required angular histogram files seem to exist in '{OUTPUT_DIR}'. Skipping preprocessing.")
        print("You can manually delete these .npy files if you wish to re-run the preprocessing.")
    else:
        print("\nOne or more angular histogram files are missing. Running preprocessing...")
        calculated_histograms = calculate_angular_histograms(
            deployment_strategies,
            metadata,
            RAW_GPS_FILE
        )
        if not calculated_histograms:
            print("\nError during preprocessing. Aborting.")
            return

    print("\n--- Preprocessing script finished ---")
    print(f"Angular histogram .npy files are saved in '{OUTPUT_DIR}'.")
    print("You can now run the optimization script (using these _accurate.npy files).")


if __name__ == "__main__":
    # 需要: pip install numpy pandas pyproj scipy
    main()
