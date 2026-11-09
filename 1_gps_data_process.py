import pandas as pd
import numpy as np
from pyproj import Transformer
import time
import os

# --- 1. 常量定义 (CONSTANTS) ---

# 输入文件路径
FILE_PATH = r"D:\科研相关存档\数据集合集\滴滴成都\GPS_WGS84_20161001.txt"

# 输出文件名
OUTPUT_VOLUME_FILE = 'data/traj_volume.npy'
OUTPUT_METADATA_FILE = 'data/grid_metadata.npy'

# 地理边界 (lon_min, lat_min, lon_max, lat_max)
BBOX_WGS = [104.0397, 30.6554, 104.12705, 30.73017]

# 坐标系
# WGS84
CRS_WGS84 = "EPSG:4326"
# 成都位于UTM Zone 48N (EPSG:32648)
CRS_UTM = "EPSG:32648"

# 网格大小（米）
GRID_SIZE_METERS = 25

# 时间窗口大小（分钟）
TIME_WINDOW_MINUTES = 5
NUM_TIMESLOTS = (24 * 60) // TIME_WINDOW_MINUTES  # 24*4 = 96


# --- 2. 主处理流程 (MAIN PROCESSING) ---

def main():
    print("--- 阶段1: 加载原始GPS数据 ---")
    start_time = time.time()

    # 定义列名
    col_names = ['vehicle_id', 'order_id', 'timestamp', 'longitude', 'latitude']

    # 加载数据
    try:
        df = pd.read_csv(
            FILE_PATH,
            header=None,
            names=col_names,
            dtype={'longitude': np.float64, 'latitude': np.float64, 'timestamp': np.int64}
        )
    except FileNotFoundError:
        print(f"错误：文件未找到 {FILE_PATH}")
        return
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return

    print(f"原始数据加载完成: {len(df)} 条记录")
    print(f"--- 阶段1耗时: {time.time() - start_time:.2f} 秒 ---")

    print("\n--- 阶段2: 时间与空间坐标转换 ---")
    start_time = time.time()

    # 1. 时间戳转换 (转换为成都 UTC+8 时间)
    #    时间戳是Unix时间戳，默认为UTC
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert('Asia/Shanghai')

    # 2. 坐标转换 (WGS84 -> UTM)
    #    always_xy=True 确保 transform 接受 (lon, lat) 顺序
    transformer_wgs_to_utm = Transformer.from_crs(CRS_WGS84, CRS_UTM, always_xy=True)
    df['x'], df['y'] = transformer_wgs_to_utm.transform(df['longitude'].values, df['latitude'].values)

    print("坐标与时间转换完成")
    print(f"--- 阶段2耗时: {time.time() - start_time:.2f} 秒 ---")

    print("\n--- 阶段3: 计算网格边界与维度 ---")

    # 1. 计算UTM边界
    transformer_bbox = Transformer.from_crs(CRS_WGS84, CRS_UTM, always_xy=True)
    min_x, min_y = transformer_bbox.transform(BBOX_WGS[0], BBOX_WGS[1])
    max_x, max_y = transformer_bbox.transform(BBOX_WGS[2], BBOX_WGS[3])

    # 2. 计算网格维度 (nx, ny)
    nx = int(np.ceil((max_x - min_x) / GRID_SIZE_METERS))
    ny = int(np.ceil((max_y - min_y) / GRID_SIZE_METERS))
    nt = NUM_TIMESLOTS

    print(f"地理边界 (UTM): X=[{min_x:.0f}, {max_x:.0f}], Y=[{min_y:.0f}, {max_y:.0f}]")
    print(f"3D立方体维度: {nx} (宽) x {ny} (高) x {nt} (时间)")

    print("\n--- 阶段4: 分箱 (Binning) 与统计 ---")
    start_time = time.time()

    # 1. 计算每个点的时间索引 k (0-95)
    df['time_k'] = (df['datetime'].dt.hour * (60 // TIME_WINDOW_MINUTES)) + \
                   (df['datetime'].dt.minute // TIME_WINDOW_MINUTES)

    # 2. 计算每个点的空间索引 i, j
    df['grid_i'] = ((df['x'] - min_x) / GRID_SIZE_METERS).astype(int)
    df['grid_j'] = ((df['y'] - min_y) / GRID_SIZE_METERS).astype(int)

    # 3. 过滤掉所有在边界之外的点
    df_filtered = df[
        (df['grid_i'] >= 0) & (df['grid_i'] < nx) &
        (df['grid_j'] >= 0) & (df['grid_j'] < ny)
        ].copy()

    print(f"过滤边界外数据后，剩余: {len(df_filtered)} 条记录")

    # 4. 使用 numpy.histogramdd 进行高效的3D分箱统计
    #    这是最快的方法，远快于 pandas.groupby()

    #   a. 准备样本数据 [N, 3]
    samples = df_filtered[['grid_i', 'grid_j', 'time_k']].values

    #   b. 定义箱子的边界
    #      np.histogramdd 需要 (N+1) 个边界
    bins_i = np.arange(nx + 1)
    bins_j = np.arange(ny + 1)
    bins_k = np.arange(nt + 1)

    #   c. 执行3D直方图统计
    TrajVolume, _ = np.histogramdd(samples, bins=[bins_i, bins_j, bins_k])

    #   d. 转换数据类型以节省空间
    TrajVolume = TrajVolume.astype(np.uint32)  # uint32 足够存储GPS点计数

    print(f"3D数据立方体构建完成")
    print(f"--- 阶段4耗时: {time.time() - start_time:.2f} 秒 ---")

    print("\n--- 阶段5: 保存结果 ---")

    # 1. 保存3D立方体
    np.save(OUTPUT_VOLUME_FILE, TrajVolume)
    print(f"成功保存3D数据立方体到: '{OUTPUT_VOLUME_FILE}'")
    print(f"  - 形状: {TrajVolume.shape} (nx, ny, nt)")
    print(f"  - 数据类型: {TrajVolume.dtype}")
    print(f"  - GPS点总数 (立方体内): {TrajVolume.sum()}")

    # 2. 保存网格元数据，这在后续计算和可视化时至关重要
    grid_metadata = {
        'nx': nx,
        'ny': ny,
        'nt': nt,
        'min_x_utm': min_x,
        'min_y_utm': min_y,
        'max_x_utm': max_x,
        'max_y_utm': max_y,
        'grid_size_meters': GRID_SIZE_METERS,
        'time_window_minutes': TIME_WINDOW_MINUTES,
        'crs_utm': CRS_UTM,
        'crs_wgs84': CRS_WGS84,
        'bbox_wgs': BBOX_WGS
    }
    np.save(OUTPUT_METADATA_FILE, grid_metadata)
    print(f"成功保存网格元数据到: '{OUTPUT_METADATA_FILE}'")

    print("\n全部处理完成！您可以运行下一个脚本了。")


if __name__ == "__main__":
    main()