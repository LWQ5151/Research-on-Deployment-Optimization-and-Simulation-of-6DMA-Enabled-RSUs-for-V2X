import numpy as np
import pandas as pd
from pyproj import Transformer
import os
import random
import time
from scipy.signal import convolve2d  # 仅用于计算 covered_potential

# --- 1. 配置参数 ---

# (输入文件)
RAW_SCORE_FILE = 'data/potential_score_raw.npy'  # DPS 热力图 (用于计算潜力)
METADATA_FILE = 'data/grid_metadata.npy'  # 网格元数据
# I-RSU 输入
I_RSU_INDEX_LIST = [1310, 2335, 434, 280, 2377, 746, 2465, 465, 797, 2255, 1170, 2242, 489, 1321, 49, 52, 105, 1599, 68,
                    1300, 1989, 2072, 847, 1045, 1386, 1756, 1882, 2251, 614, 664, 848, 194]  # 取前 32 个
I_RSU_LOC_FILE = 'deploy_results/street_count_origin_edt.txt'
# Traffic-Greedy 输入
TRAFFIC_INDEX_LIST = [1252, 1139, 858, 578, 677, 1302, 1014, 100, 375, 346, 523, 831, 887, 136, 1571, 768, 356, 158,
                      886, 105, 647, 1104, 112, 1562, 970, 1410, 787, 267, 81, 1158, 150, 50]  # 取前 32 个
TRAFFIC_LOC_FILE = 'deploy_results/intersection_traffic_density_300.txt'

# (输出目录)
OUTPUT_DIR = 'deploy_results/'

# (部署参数)
N_RSUS = 32
RSU_COVERAGE_RADIUS_METERS = 500.0


# --- 2. 辅助函数 ---

def create_circular_kernel(radius_in_grids):
    size = int(np.ceil(radius_in_grids) * 2 + 1)
    hf = size // 2
    ii, jj = np.indices((size, size))
    dist = np.sqrt((ii - hf) ** 2 + (jj - hf) ** 2)
    kernel = (dist <= radius_in_grids).astype(float)
    return kernel


def grid_to_utm(i, j, metadata):
    x = metadata['min_x_utm'] + (i + 0.5) * metadata['grid_size_meters']
    y = metadata['min_y_utm'] + (j + 0.5) * metadata['grid_size_meters']
    return x, y


def calculate_covered_score(heatmap, center_i, center_j, radius_in_grids):
    nx, ny = heatmap.shape
    ii, jj = np.indices((nx, ny))
    dist = np.sqrt((ii - center_i) ** 2 + (jj - center_j) ** 2)
    mask = (dist <= radius_in_grids)
    # 确保掩码内的索引有效
    valid_mask = mask & (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
    return np.sum(heatmap[valid_mask])


# --- 3. Baseline 实现 ---

def baseline_traffic_from_indices(dps_heatmap, metadata, N):
    """
    Baseline: Traffic-Greedy (使用预计算索引)
    """
    print(f"\n--- Baseline: Running Traffic Deployment (from indices, N={N}) ---")
    start_time = time.time()
    results = []
    grid_size = metadata['grid_size_meters']
    radius_in_grids = RSU_COVERAGE_RADIUS_METERS / grid_size

    try:
        # 加载包含位置信息的文件
        # 【重要假设】假设此文件包含 'lon', 'lat' 列
        traffic_loc_df = pd.read_csv(TRAFFIC_LOC_FILE, delimiter=';')  # 注意分隔符

        # 验证必需的列是否存在
        if 'lon' not in traffic_loc_df.columns or 'lat' not in traffic_loc_df.columns:
            print(f"错误: 文件 '{TRAFFIC_LOC_FILE}' 缺少 'lon' 或 'lat' 列。")
            print("将尝试使用 'dx', 'dy' 作为网格索引 (如果存在)...")
            if 'dx' not in traffic_loc_df.columns or 'dy' not in traffic_loc_df.columns:
                print(f"错误: 文件 '{TRAFFIC_LOC_FILE}' 也缺少 'dx' 或 'dy' 列。无法继续。")
                return None
            else:
                print("警告: 正在使用 'dx', 'dy' 作为网格索引。请确保这是正确的映射！")
                use_dx_dy = True
        else:
            use_dx_dy = False

        if len(TRAFFIC_INDEX_LIST) < N:
            print(f"Warning: Only {len(TRAFFIC_INDEX_LIST)} Traffic indices provided, requested {N}. Using available.")
            indices_to_use = TRAFFIC_INDEX_LIST
        else:
            indices_to_use = TRAFFIC_INDEX_LIST[:N]

        crs_wgs84 = metadata.get('crs_wgs84', "EPSG:4326")
        crs_utm = metadata.get('crs_utm', "EPSG:32648")
        transformer_wgs_to_utm = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)

        min_x_utm = metadata['min_x_utm']
        min_y_utm = metadata['min_y_utm']
        nx = metadata['nx']
        ny = metadata['ny']

        selected_grids = set()
        count = 0

        for idx in indices_to_use:
            if count >= N: break

            try:
                loc_data = traffic_loc_df.iloc[idx]
            except IndexError:
                print(f"警告: 索引 {idx} 超出文件 '{TRAFFIC_LOC_FILE}' 的范围，跳过。")
                continue

            if use_dx_dy:
                # 使用 dx, dy 作为 grid_i, grid_j (高风险假设)
                # 注意：您的代码用的是 (dy, dx)，这里遵循它
                grid_i = int(loc_data['dy'])  # 假设 dy 对应 i (x 轴)
                grid_j = int(loc_data['dx'])  # 假设 dx 对应 j (y 轴)
                # 需要从 grid 反推 UTM (这需要 grid_to_utm 支持)
                utm_x, utm_y = grid_to_utm(grid_i, grid_j, metadata)  # 假设 grid_to_utm 可以工作
            else:
                # 使用 lon, lat
                lon = loc_data['lon']
                lat = loc_data['lat']
                utm_x, utm_y = transformer_wgs_to_utm.transform(lon, lat)
                grid_i = int((utm_x - min_x_utm) / grid_size)
                grid_j = int((utm_y - min_y_utm) / grid_size)

            grid_i = max(0, min(nx - 1, grid_i))
            grid_j = max(0, min(ny - 1, grid_j))

            if (grid_i, grid_j) in selected_grids:
                continue

            covered_dps = calculate_covered_score(dps_heatmap, grid_i, grid_j, radius_in_grids)

            results.append({
                'id': count + 1,
                'grid_i': int(grid_i),
                'grid_j': int(grid_j),
                'utm_x': utm_x,
                'utm_y': utm_y,
                'covered_potential': covered_dps,  # 计算实际覆盖的 DPS
            })
            selected_grids.add((grid_i, grid_j))
            count += 1

        print(f"Processed {count} unique Traffic locations.")

    except FileNotFoundError:
        print(f"Error: Traffic locations file not found at {TRAFFIC_LOC_FILE}")
        return None
    except Exception as e:
        print(f"Error processing Traffic file: {e}")
        return None

    df = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, 'deployment_traffic_greedy.csv')  # 文件名保持一致
    df.to_csv(output_file, index=False)
    print(f"Traffic results (from indices) saved to {output_file}")
    print(f"--- Traffic (from indices) completed in {time.time() - start_time:.2f} seconds ---")
    return df


def baseline_i_rsu(dps_heatmap, metadata, N):
    """
    Baseline: I-RSU (使用预计算索引)
    """
    print(f"\n--- Baseline: Running I-RSU Deployment (from indices, N={N}) ---")
    start_time = time.time()
    results = []
    grid_size = metadata['grid_size_meters']
    radius_in_grids = RSU_COVERAGE_RADIUS_METERS / grid_size

    try:
        # 加载包含经纬度信息的文件
        i_rsu_loc_df = pd.read_csv(I_RSU_LOC_FILE, delimiter=',')  # 注意分隔符
        # 验证列名
        if 'lon' not in i_rsu_loc_df.columns or 'lat' not in i_rsu_loc_df.columns:
            print(f"错误: 文件 '{I_RSU_LOC_FILE}' 缺少 'lon' 或 'lat' 列。")
            return None

        if len(I_RSU_INDEX_LIST) < N:
            print(f"Warning: Only {len(I_RSU_INDEX_LIST)} I-RSU indices provided, requested {N}. Using available.")
            indices_to_use = I_RSU_INDEX_LIST
        else:
            indices_to_use = I_RSU_INDEX_LIST[:N]

        crs_wgs84 = metadata.get('crs_wgs84', "EPSG:4326")
        crs_utm = metadata.get('crs_utm', "EPSG:32648")
        transformer_wgs_to_utm = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)

        min_x_utm = metadata['min_x_utm']
        min_y_utm = metadata['min_y_utm']
        nx = metadata['nx']
        ny = metadata['ny']

        selected_grids = set()
        count = 0

        for idx in indices_to_use:
            if count >= N: break

            try:
                # 使用索引从 DataFrame 获取经纬度
                loc_data = i_rsu_loc_df.iloc[idx]
                lon = loc_data['lon']
                lat = loc_data['lat']
            except IndexError:
                print(f"警告: 索引 {idx} 超出文件 '{I_RSU_LOC_FILE}' 的范围，跳过。")
                continue
            except KeyError as e:
                print(f"警告: 文件 '{I_RSU_LOC_FILE}' 缺少列 {e}，跳过索引 {idx}。")
                continue

            utm_x, utm_y = transformer_wgs_to_utm.transform(lon, lat)
            grid_i = int((utm_x - min_x_utm) / grid_size)
            grid_j = int((utm_y - min_y_utm) / grid_size)
            grid_i = max(0, min(nx - 1, grid_i))
            grid_j = max(0, min(ny - 1, grid_j))

            if (grid_i, grid_j) in selected_grids:
                continue

            covered_dps = calculate_covered_score(dps_heatmap, grid_i, grid_j, radius_in_grids)

            results.append({
                'id': count + 1,
                'grid_i': int(grid_i),
                'grid_j': int(grid_j),
                'utm_x': utm_x,
                'utm_y': utm_y,
                'covered_potential': covered_dps,
                'original_lon': lon,
                'original_lat': lat
            })
            selected_grids.add((grid_i, grid_j))
            count += 1

        print(f"Processed {count} unique I-RSU locations.")

    except FileNotFoundError:
        print(f"Error: I-RSU locations file not found at {I_RSU_LOC_FILE}")
        return None
    except Exception as e:
        print(f"Error processing I-RSU file: {e}")
        return None

    df = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, 'deployment_i_rsu.csv')  # 文件名保持一致
    df.to_csv(output_file, index=False)
    print(f"I-RSU results (from indices) saved to {output_file}")
    print(f"--- I-RSU (from indices) completed in {time.time() - start_time:.2f} seconds ---")
    return df


def baseline_random(dps_heatmap, metadata, N):
    """
    Baseline: Random
    随机选择 N 个不同的网格位置，并计算它们覆盖的 DPS。
    """
    print(f"\n--- Baseline: Running Random Deployment (N={N}) ---")
    # ... (此函数逻辑保持不变) ...
    start_time = time.time()
    nx = metadata['nx']
    ny = metadata['ny']
    grid_size = metadata['grid_size_meters']
    radius_in_grids = RSU_COVERAGE_RADIUS_METERS / grid_size

    all_indices = [(i, j) for i in range(nx) for j in range(ny)]

    if len(all_indices) < N:
        print(f"Warning: Number of grid cells ({len(all_indices)}) is less than requested N ({N}). Using all cells.")
        selected_indices = all_indices
        N = len(all_indices)
    else:
        selected_indices = random.sample(all_indices, N)

    print(f"Selected {N} random locations.")

    results = []
    for k in range(N):
        i_best, j_best = selected_indices[k]
        utm_x, utm_y = grid_to_utm(i_best, j_best, metadata)
        covered_dps = calculate_covered_score(dps_heatmap, i_best, j_best, radius_in_grids)

        results.append({
            'id': k + 1,
            'grid_i': i_best,
            'grid_j': j_best,
            'utm_x': utm_x,
            'utm_y': utm_y,
            'covered_potential': covered_dps
        })

    df = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, 'deployment_random.csv')  # 文件名保持一致
    df.to_csv(output_file, index=False)
    print(f"Random results saved to {output_file}")
    print(f"--- Random completed in {time.time() - start_time:.2f} seconds ---")
    return df


# --- 4. 主执行函数 ---

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading core data files...")
    try:
        dps_heatmap = np.load(RAW_SCORE_FILE)
        # traj_volume = np.load(TRAJ_VOLUME_FILE) # 不再需要 Traffic-Greedy 的体积数据
        metadata = np.load(METADATA_FILE, allow_pickle=True).item()
    except FileNotFoundError as e:
        print(f"Error: Cannot load input file {e.filename}.")
        return
    print("Data loaded successfully.")

    # 运行修改后的 Baselines
    # baseline_dps_topn(dps_heatmap, metadata, N_RSUS) # 已移除
    baseline_traffic_from_indices(dps_heatmap, metadata, N_RSUS)  # 使用新函数名
    baseline_i_rsu(dps_heatmap, metadata, N_RSUS)
    baseline_random(dps_heatmap, metadata, N_RSUS)

    print("\n--- All baseline deployment strategies executed. ---")


if __name__ == "__main__":
    main()
