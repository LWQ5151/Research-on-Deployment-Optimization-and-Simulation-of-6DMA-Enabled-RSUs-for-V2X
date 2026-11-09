import numpy as np
import pandas as pd
from scipy.signal import convolve2d
import folium
from folium.plugins import HeatMap
from pyproj import Transformer
import os
import csv
import time

# --- 1. 配置参数 ---

# (输入文件 - 必须已存在)
RAW_SCORE_FILE = 'data/potential_score_raw.npy'
METADATA_FILE = 'data/grid_metadata.npy'

# (输出文件 - 将由本脚本生成)
DEPLOYMENT_CSV_FILE = 'results/deployment_strategy.csv'
OUTPUT_MAP_HTML = 'results/deployment_on_map.html'  # 最终的地图 (HTML)

# (部署参数 - 您可以在此调整)
N_RSUS = 32  # 您希望部署的RSU总数
RSU_COVERAGE_RADIUS_METERS = 500  # 6DMA-RSU的覆盖半径(米)

# (Folium 可视化参数)
HEATMAP_BLUR = 10
HEATMAP_RADIUS = 15
HEATMAP_MAX_ZOOM = 12


# --- 2. 辅助函数 (来自阶段1) ---
# (这部分与 V11 脚本相同)

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


def mask_covered_area(heatmap, center_i, center_j, radius_in_grids):
    nx, ny = heatmap.shape
    ii, jj = np.indices((nx, ny))
    dist = np.sqrt((ii - center_i) ** 2 + (jj - center_j) ** 2)
    mask = (dist <= radius_in_grids)
    heatmap[mask] = 0
    return heatmap


# --- 3. 核心部署函数 (阶段1) ---
# (这部分与 V11 脚本相同)

def greedy_deployment(potential_heatmap, metadata, N, radius_meters):
    print(f"--- 阶段1: 开始贪心部署 {N} 个RSU ---")

    nx, ny = potential_heatmap.shape
    grid_size = metadata['grid_size_meters']
    radius_in_grids = radius_meters / grid_size

    print(f"RSU覆盖半径: {radius_meters} 米 (约 {radius_in_grids:.1f} 个网格)")

    scoring_kernel = create_circular_kernel(radius_in_grids)
    deployed_rsus = []

    for k in range(N):
        start_iter = time.time()

        score_map = convolve2d(potential_heatmap, scoring_kernel, mode='same')

        i_best, j_best = np.unravel_index(np.argmax(score_map), score_map.shape)
        max_score = score_map[i_best, j_best]

        if max_score < 1e-9:
            print(f"地图潜力已耗尽，在部署第 {k + 1} 个RSU时提前停止。")
            break

        utm_x, utm_y = grid_to_utm(i_best, j_best, metadata)
        rsu_info = {
            'id': k + 1,
            'grid_i': i_best,
            'grid_j': j_best,
            'utm_x': utm_x,
            'utm_y': utm_y,
            'covered_potential': max_score
        }
        deployed_rsus.append(rsu_info)

        potential_heatmap = mask_covered_area(potential_heatmap,
                                              i_best, j_best,
                                              radius_in_grids)

        print(f"  部署 RSU #{k + 1}/{N}: "
              f"位置 (i={i_best}, j={j_best}), "
              f"得分={max_score:.0f}, "
              f"耗时 {time.time() - start_iter:.2f} 秒")

    print(f"--- 贪心部署完成，共部署 {len(deployed_rsus)} 个RSU ---")
    return deployed_rsus


# --- 4. 核心Folium可视化函数 (阶段2 - V12版) ---
# (此函数替换 V11 脚本中的同名函数)

def visualize_deployment_on_folium(heatmap_original, deployed_rsus_df, metadata, radius_meters):
    """
    使用 Folium 将热力图和部署点叠加到HTML地图上
    [V12 修正] 更改为适合英文期刊的、无标签的浅色底图
    """
    print(f"\n--- 阶段2: 开始 Folium 地理可视化 ---")

    # 1. 准备坐标转换器
    crs_utm = metadata['crs_utm']
    crs_wgs84 = metadata['crs_wgs84']
    transformer_utm_to_wgs = Transformer.from_crs(crs_utm, crs_wgs84, always_xy=True)

    # 2. 准备 Folium 热力图数据 [lat, lon, weight]
    print("正在准备热力图数据 (转换坐标)...")

    # a. 找到所有非零潜力的网格
    nonzero_indices = np.nonzero(heatmap_original)
    scores = heatmap_original[nonzero_indices]

    # b. (可选) 过滤，只显示top 5%的潜力点
    percentile_95 = np.percentile(scores, 95)
    high_potential_mask = (scores >= percentile_95)

    grid_i = nonzero_indices[0][high_potential_mask]
    grid_j = nonzero_indices[1][high_potential_mask]
    scores_filtered = scores[high_potential_mask]

    print(f"  (过滤非零点，只显示Top 5%的潜力点: {len(scores_filtered)} 个)")

    # c. 将网格索引 (i, j) 批量转换为 UTM (x, y)
    grid_size = metadata['grid_size_meters']
    min_x_utm = metadata['min_x_utm']
    min_y_utm = metadata['min_y_utm']
    utm_x = min_x_utm + (grid_i + 0.5) * grid_size
    utm_y = min_y_utm + (grid_j + 0.5) * grid_size

    # d. 将 UTM (x, y) 批量转换为 WGS84 (lon, lat)
    lon, lat = transformer_utm_to_wgs.transform(utm_x, utm_y)

    # e. 归一化分数 (HeatMap 插件需要)
    scores_normalized = (scores_filtered - scores_filtered.min()) / \
                        (scores_filtered.max() - scores_filtered.min() + 1e-9)

    # f. 创建最终数据列表
    heatmap_data = list(zip(lat, lon, scores_normalized))

    # 3. 准备 RSU 部署点数据 (WGS84)
    print("正在准备 RSU 部署点 (转换坐标)...")
    rsus_lon, rsus_lat = transformer_utm_to_wgs.transform(
        deployed_rsus_df['utm_x'].values,
        deployed_rsus_df['utm_y'].values
    )
    deployed_rsus_df['longitude'] = rsus_lon
    deployed_rsus_df['latitude'] = rsus_lat

    # 4. 创建 Folium 地图
    bbox_wgs = metadata['bbox_wgs']  # [lon_min, lat_min, lon_max, lat_max]
    center_lon = (bbox_wgs[0] + bbox_wgs[2]) / 2
    center_lat = (bbox_wgs[1] + bbox_wgs[3]) / 2

    print(f"正在创建 Folium 地图，中心点: ({center_lat:.4f}, {center_lon:.4f})")

    # **********************************************
    # * 关键修改在这里 (V12)
    # **********************************************
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="CartoDB.PositronNoLabels"  # <-- 从 "cartodbdarkmatter" 修改
    )
    # (备选方案: 如果您想要纯黑白的，可以使用 "Stamen.TonerLite")
    # **********************************************

    # 5. 添加热力图层
    HeatMap(
        heatmap_data,
        name='RSU Deployment Potential (Top 5%)',  # <-- 修改为英文
        blur=HEATMAP_BLUR,
        radius=HEATMAP_RADIUS,
        max_zoom=HEATMAP_MAX_ZOOM
    ).add_to(m)

    # 6. 添加 RSU 部署层
    rsu_layer = folium.FeatureGroup(name=f"{N_RSUS} RSU Deployments", show=True)  # <-- 修改为英文

    for _, rsu in deployed_rsus_df.iterrows():
        popup_html = (
            f"<b>RSU ID: {rsu['id']}</b><br>"
            f"UTM (x,y): ({rsu['utm_x']:.0f}, {rsu['utm_y']:.0f})<br>"
            f"Covered Potential: {rsu['covered_potential']:.0f}"  # <-- 修改为英文
        )

        # **********************************************
        # * 关键修改在这里 (V13)
        # **********************************************

        # a. 绘制 RSU 中心点 (使用 CircleMarker 替代 Marker)
        folium.CircleMarker(
            location=[rsu['latitude'], rsu['longitude']],
            radius=3,  # 半径为3个像素 (一个小红点)
            color="#3186cc",
            fill=True,
            fill_color="#3186cc",
            fill_opacity=1.0,  # 实心红点
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"RSU ID: {rsu['id']}"  # 鼠标悬停提示
        ).add_to(rsu_layer)

        # **********************************************

        # b. 绘制 RSU 覆盖圈
        folium.Circle(
            location=[rsu['latitude'], rsu['longitude']],
            radius=radius_meters,
            color="#3186cc",
            weight=1,
            fill=True,
            fill_color="#3186cc",
            fill_opacity=0.1
        ).add_to(rsu_layer)

    rsu_layer.add_to(m)

    # 7. 添加图层控制器
    folium.LayerControl().add_to(m)

    # 8. 保存地图到 HTML
    print(f"正在保存最终地图到 {OUTPUT_MAP_HTML}...")
    m.save(OUTPUT_MAP_HTML)

    print("\n--- Folium 地理可视化成功! ---")
    print(f"部署地图已保存到: {OUTPUT_MAP_HTML}")
    print(f"请在您的浏览器中打开 '{OUTPUT_MAP_HTML}' 查看结果。")


# --- 5. 主执行函数 ---

if __name__ == "__main__":
    # 0. 检查依赖
    try:
        import folium
        import pyproj
        import scipy
    except ImportError as e:
        print(f"错误：缺少必要的库: {e}")
        print("请运行: pip install numpy pandas scipy folium pyproj")
        exit()

    # 1. 加载输入数据
    if not all(os.path.exists(f) for f in [RAW_SCORE_FILE, METADATA_FILE]):
        print("错误：缺少一个或多个输入文件。")
        print(f"请确保 '{RAW_SCORE_FILE}' 和 '{METADATA_FILE}' 都存在。")
        print("(您需要先运行潜力计算脚本 V10 或最终版。)")
    else:
        print("正在加载输入数据...")
        heatmap_original = np.load(RAW_SCORE_FILE)
        metadata = np.load(METADATA_FILE, allow_pickle=True).item()

        # 2. 执行阶段1: 贪心部署
        deployed_rsus = greedy_deployment(
            heatmap_original.copy(),
            metadata,
            N_RSUS,
            RSU_COVERAGE_RADIUS_METERS
        )

        # 3. 保存部署策略到 CSV
        if deployed_rsus:
            print(f"\n正在保存部署策略到 '{DEPLOYMENT_CSV_FILE}'...")
            try:
                with open(DEPLOYMENT_CSV_FILE, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=deployed_rsus[0].keys())
                    writer.writeheader()
                    writer.writerows(deployed_rsus)
            except Exception as e:
                print(f"保存 CSV 失败: {e}")
                exit()

            # 4. 执行阶段2: 地理可视化
            deployed_rsus_df = pd.DataFrame(deployed_rsus)
            visualize_deployment_on_folium(
                heatmap_original,
                deployed_rsus_df,
                metadata,
                RSU_COVERAGE_RADIUS_METERS
            )
        else:
            print("未部署任何 RSU，跳过保存和可视化。")

    print("\n--- 统一脚本执行完毕 ---")