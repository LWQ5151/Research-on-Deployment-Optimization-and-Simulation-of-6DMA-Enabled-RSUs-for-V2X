import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
# import contextily as cx # 不再需要 contextily
import os
import matplotlib
import matplotlib.ticker as mticker
from pyproj import Transformer

# --- 设置字体大小 ---
matplotlib.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

# --- 1. 配置参数 ---

# (输入文件 - 必须已存在)
METADATA_FILE = 'data/grid_metadata.npy'
OUTPUT_DIR = 'deploy_results/' # Baseline CSV 文件所在目录
# 包含我们自己的方案
DEPLOYMENT_FILES = {
    'DPS-Greedy (Ours)': os.path.join(OUTPUT_DIR, 'deployment_dps_greedy.csv'),
    'Traffic-Greedy': os.path.join(OUTPUT_DIR, 'deployment_traffic_greedy.csv'),
    'I-RSU': os.path.join(OUTPUT_DIR, 'deployment_i_rsu.csv'),
    'Random': os.path.join(OUTPUT_DIR, 'deployment_random.csv')
}
PLOT_ORDER = ['DPS-Greedy (Ours)', 'Traffic-Greedy', 'I-RSU', 'Random']

# [修改] 指定离线 Shapefile 底图路径
SHAPEFILE_PATH = 'data/shapefiles/edges.shp' # <--- 请将这里替换为您 Shapefile 文件的实际路径

# (输出文件)
OUTPUT_PDF_FILE = 'results/baseline_deployments_comparison_shapefile_latlon.pdf' # 新文件名

# (部署参数)
RSU_COVERAGE_RADIUS_METERS = 500.0

# (绘图颜色)
COLORS = ['#3186cc', '#ff7f0e', '#2ca02c', '#9467bd']
SHAPEFILE_COLOR = 'gray' # 底图线条颜色
SHAPEFILE_LINEWIDTH = 0.5 # 底图线条宽度


# --- 2. 辅助绘图函数 ---

def plot_single_baseline_with_shapefile(ax, deployment_df, metadata, title, color, gdf_background):
    """
    在给定的 matplotlib Axes 上绘制单个 baseline 的部署方案 (带 Shapefile 底图, 经纬度)。
    gdf_background: 预先加载并转换好坐标系的 Shapefile GeoDataFrame
    """
    crs_utm = metadata['crs_utm']
    crs_wgs84 = metadata['crs_wgs84']
    target_crs = gdf_background.crs # 获取底图的坐标系，所有绘图都统一到这个坐标系

    # 1. 创建 RSU 点的 GeoDataFrame (初始为 UTM)
    try:
        gdf_rsus_utm = gpd.GeoDataFrame(
            deployment_df,
            geometry=gpd.points_from_xy(deployment_df.utm_x, deployment_df.utm_y),
            crs=crs_utm
        )
    except Exception as e:
        print(f"Error creating GeoDataFrame for {title}: {e}")
        ax.set_title(f"{title}\n(Error Loading Data)", fontsize=12)
        return False, None

    # 2. 创建 RSU 覆盖圈 GeoDataFrame (在 UTM 下计算 buffer)
    gdf_buffers_utm = gdf_rsus_utm.copy()
    gdf_buffers_utm['geometry'] = gdf_buffers_utm.geometry.buffer(RSU_COVERAGE_RADIUS_METERS)

    # 3. 将 RSU 数据转换为与底图相同的坐标系
    print(f"  Converting RSU geometries to background CRS ({target_crs.to_string()})...")
    try:
        gdf_rsus = gdf_rsus_utm.to_crs(target_crs)
        gdf_buffers = gdf_buffers_utm.to_crs(target_crs)
    except Exception as e:
        print(f"  Error converting RSU CRS for {title}: {e}")
        ax.set_title(f"{title}\n(CRS Conversion Error)", fontsize=12)
        return False, None

    # 4. [修改] 绘制 Shapefile 底图 (最底层)
    print(f"  Plotting background shapefile for {title}...")
    try:
        gdf_background.plot(
            ax=ax,
            color=SHAPEFILE_COLOR,
            linewidth=SHAPEFILE_LINEWIDTH,
            zorder=1, # 最底层
            alpha = 0.3 #透明度
        )
    except Exception as e:
        print(f"  Warning: Could not plot background shapefile for {title}: {e}")

    # 5. 绘制覆盖圈 (中层)
    gdf_buffers.plot(
        ax=ax,
        edgecolor=color,
        facecolor=color,
        alpha=0.1,
        linestyle='-',
        linewidth=0.5,
        zorder=2 # 在底图之上
    )

    # 6. 绘制 RSU 中心点 (顶层)
    gdf_rsus.plot(
        ax=ax,
        marker='o',
        color=color,
        markersize=15,
        linewidth=0,
        zorder=3 # 在覆盖圈之上
    )

    # 7. 设置子图属性
    ax.set_title(title, fontsize=16)
    ax.set_aspect('equal', adjustable='box') # 保持地理比例
    # 隐藏坐标轴刻度标签，底图提供参照
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # 返回转换后坐标系的边界，用于统一视图
    return True, gdf_buffers.total_bounds

# --- 3. 主执行函数 ---

def main():
    # 0. 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_PDF_FILE), exist_ok=True)

    # 1. 加载元数据
    print("Loading metadata...")
    try:
        metadata = np.load(METADATA_FILE, allow_pickle=True).item()
        # 检查所需元数据
        if 'crs_utm' not in metadata or 'crs_wgs84' not in metadata or 'bbox_wgs' not in metadata:
            print("Error: Metadata file is missing required CRS or BBox information.")
            return
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return

    # 2. [修改] 加载离线 Shapefile 底图
    print(f"Loading background shapefile from: {SHAPEFILE_PATH}")
    try:
        gdf_background_full = gpd.read_file(SHAPEFILE_PATH)
        # 可选：如果 Shapefile 很大，可以裁剪到研究区域附近以提高绘图速度
        # 例如，先获取研究区域的 WGS84 边界
        bbox_wgs = metadata['bbox_wgs']
        min_lon, min_lat, max_lon, max_lat = bbox_wgs
        # 创建一个边界框多边形 (需要 shapely)
        # from shapely.geometry import box
        # bbox_poly = box(min_lon-0.01, min_lat-0.01, max_lon+0.01, max_lat+0.01) # 稍微扩大
        # gdf_background = gdf_background_full.clip(bbox_poly)
        # print(f"  Clipped background shapefile to study area.")

        # 为简单起见，我们先使用完整的 Shapefile
        gdf_background = gdf_background_full

        # 将底图转换为 WGS84 (如果需要的话)，以便后续绘图
        # Folium 和 contextily 通常使用 Web Mercator (EPSG:3857) 或 WGS84 (EPSG:4326)
        # Matplotlib 可以处理多种 CRS，但统一到 WGS84 较方便
        target_crs_wgs = metadata['crs_wgs84']
        if gdf_background.crs != target_crs_wgs:
             print(f"  Converting background shapefile CRS from {gdf_background.crs} to {target_crs_wgs}...")
             gdf_background = gdf_background.to_crs(target_crs_wgs)

    except FileNotFoundError:
        print(f"Error: Background shapefile not found at {SHAPEFILE_PATH}")
        print("Cannot plot with background map.")
        return
    except Exception as e:
        print(f"Error loading or processing shapefile: {e}")
        return


    # 3. 创建图窗和子图 (1行4列)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle('Comparison of RSU Deployment Strategies (N=32)', fontsize=20, y=1.03)

    plot_idx = 0
    all_bounds = [] # 存储每个子图的边界 (现在是底图坐标系, WGS84)
    plot_success_flags = []

    # 4. 循环绘制每个策略
    for i, key in enumerate(PLOT_ORDER):
        filepath = DEPLOYMENT_FILES.get(key)
        ax = axes[plot_idx]
        color = COLORS[i % len(COLORS)]

        if filepath and os.path.exists(filepath):
            print(f"\nProcessing: {key} ({filepath})")
            try:
                deployment_df = pd.read_csv(filepath)
                if not deployment_df.empty:
                    # [修改] 调用新的绘图函数
                    success, bounds = plot_single_baseline_with_shapefile(ax, deployment_df, metadata, key, color, gdf_background)
                    plot_success_flags.append(success)
                    if success and bounds is not None:
                        all_bounds.append(bounds)
                else:
                    print(f"Warning: Deployment file '{filepath}' is empty.")
                    ax.set_title(f"{key}\n(No Data)", fontsize=12)
                    ax.axis('off')
                    plot_success_flags.append(False)
            except Exception as e:
                print(f"Error processing or plotting {key}: {e}")
                ax.set_title(f"{key}\n(Plotting Error)", fontsize=12)
                ax.axis('off')
                plot_success_flags.append(False)
        else:
            print(f"Warning: Baseline file for '{key}' not found at '{filepath}'. Skipping.")
            ax.set_title(f"{key}\n(File Not Found)", fontsize=12)
            ax.axis('off')
            plot_success_flags.append(False)

        plot_idx += 1

    # 5. 统一坐标轴范围 (使用 WGS84 边界)
    if all(plot_success_flags) and all_bounds:
        print("\nSetting unified axis limits based on WGS84 bounds...")
        # (minx, miny, maxx, maxy)
        global_min_lon = min(b[0] for b in all_bounds) - 0.005
        global_min_lat = min(b[1] for b in all_bounds) - 0.005
        global_max_lon = max(b[2] for b in all_bounds) + 0.005
        global_max_lat = max(b[3] for b in all_bounds) + 0.005

        for ax in axes:
            if ax.axison:
                ax.set_xlim(global_min_lon, global_max_lon)
                ax.set_ylim(global_min_lat, global_max_lat)

    # 6. 调整布局并保存 PDF
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    try:
        plt.savefig(OUTPUT_PDF_FILE, format='pdf', bbox_inches='tight', dpi=300)
        print(f"\n--- Comparison visualization saved successfully! ---")
        print(f"PDF saved to: {OUTPUT_PDF_FILE}")
    except Exception as e:
        print(f"\nError saving PDF file: {e}")

    plt.show()

if __name__ == "__main__":
    main()
