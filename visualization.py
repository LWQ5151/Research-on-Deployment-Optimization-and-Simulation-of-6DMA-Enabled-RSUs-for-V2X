import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # 导入对数刻度
import os
import matplotlib
import matplotlib.ticker as mticker # [FIX] Import ticker

# --- 设置字体大小 ---
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# --- 1. 定义文件名 ---

HEATMAP_FILE = 'data/potential_score_raw.npy' # 使用原始分值
METADATA_FILE = 'data/grid_metadata.npy'
OUTPUT_IMAGE_FILE = 'fig/potential_heatmap_latlon.pdf'  # 输出为 PDF (新文件名)

# --- 2. 主可视化函数 (经纬度 + 对比度优化版) ---

def visualize_heatmap():
    print(f"Starting visualization (Lat/Lon Enhanced Contrast PDF Version)...")

    # 1. 检查文件是否存在
    if not os.path.exists(HEATMAP_FILE) or not os.path.exists(METADATA_FILE):
        print(f"Error: Missing required files.")
        print(f"Ensure '{HEATMAP_FILE}' and '{METADATA_FILE}' exist.")
        return

    # 2. 加载数据
    try:
        heatmap_raw = np.load(HEATMAP_FILE)
        metadata = np.load(METADATA_FILE, allow_pickle=True).item()
        print(f"Successfully loaded raw heatmap, shape: {heatmap_raw.shape}")

        # 找到非零值用于 LogNorm 和对比度调整
        scores_nonzero = heatmap_raw[heatmap_raw > 0]
        if len(scores_nonzero) == 0:
            print("Warning: No non-zero scores found in heatmap.")
            vmin_log = 1e-9
            vmax_log = 1.0 # 设置默认值
        else:
            vmin_log = np.min(scores_nonzero)
            # [优化对比度] 使用 99.9 百分位作为 vmax，使最高分的颜色更突出
            vmax_log = np.percentile(scores_nonzero, 99.9)
            # 如果 vmax_log 异常地小 (例如接近 vmin_log)，则回退到使用最大值
            if vmax_log <= vmin_log * 1.1:
                vmax_log = scores_nonzero.max()
        print(f"LogNorm range: vmin={vmin_log:.2e}, vmax={vmax_log:.2e} (using 99.9th percentile)")

    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 3. 提取元数据
    nx = metadata.get('nx')
    ny = metadata.get('ny')
    bbox_wgs = metadata.get('bbox_wgs') # [lon_min, lat_min, lon_max, lat_max]
    if bbox_wgs is None:
        print("Error: Metadata file missing 'bbox_wgs'. Cannot use Lat/Lon.")
        return
    lon_min, lat_min, lon_max, lat_max = bbox_wgs

    if any(v is None for v in [nx, ny]):
        print("Error: Metadata file is incomplete (missing nx or ny).")
        return

    # 4. 准备绘图数据 (转置)
    if heatmap_raw.shape == (nx, ny):
        heatmap_to_plot = heatmap_raw.T
        print(f"Heatmap shape (nx, ny) -> {heatmap_raw.shape}, transposing to (ny, nx) -> {heatmap_to_plot.shape} for plotting.")
    elif heatmap_raw.shape == (ny, nx):
         heatmap_to_plot = heatmap_raw
         print(f"Heatmap shape (ny, nx) -> {heatmap_raw.shape}, plotting directly.")
    else:
        print(f"Warning: Heatmap shape {heatmap_raw.shape} mismatch with metadata {nx}x{ny}.")
        return

    # 5. 计算经纬度范围的宽高比 (近似)
    delta_lon = lon_max - lon_min
    delta_lat = lat_max - lat_min
    center_lat_rad = np.radians((lat_min + lat_max) / 2)
    # Correct aspect ratio calculation (height/width)
    aspect_ratio = (delta_lat * 111) / (delta_lon * 111 * np.cos(center_lat_rad)) if delta_lon > 0 else 1

    # --- 6. 开始绘图 ---
    print("Generating plot...")

    fig_width = 8
    # Ensure fig_height is positive
    fig_height = max(1, fig_width * aspect_ratio) + 1.0

    plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca() # Get current axes

    # 使用 imshow 绘制热力图
    im = ax.imshow(
        heatmap_to_plot,
        cmap='viridis',     # [修改] 改回 viridis
        origin='lower',
        extent=[lon_min, lon_max, lat_min, lat_max], # 使用经纬度范围
        interpolation='nearest',
        norm=LogNorm(vmin=vmin_log, vmax=vmax_log) # 使用对数刻度和调整后的vmax
    )

    # 7. 设置图表属性 (英文, 经纬度)
    ax.set_title("RSU Dynamic Potential Score (DPS) Visualization")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # 尝试保持正确的地理宽高比
    ax.set_aspect('equal', adjustable='box')

    # 添加色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03, label='Dynamic Potential Score (DPS) - Log Scale')

    # 调整刻度标签大小
    ax.tick_params(axis='both', which='major', labelsize=10)

    # [FIX] 使用 FormatStrFormatter 格式化经纬度坐标轴标签
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))

    plt.xticks(rotation=30, ha='right') # 旋转X轴标签防止重叠, ha='right' improves alignment
    cbar.ax.tick_params(labelsize=10)


    # 8. 保存图像 (PDF)
    # [FIX] Call tight_layout BEFORE savefig for better results
    plt.tight_layout(pad=1.5)
    try:
        os.makedirs(os.path.dirname(OUTPUT_IMAGE_FILE), exist_ok=True)
        plt.savefig(OUTPUT_IMAGE_FILE, format='pdf', bbox_inches='tight')
        print(f"\n--- Visualization successful! ---")
        print(f"Heatmap saved to: {OUTPUT_IMAGE_FILE}")
    except Exception as e:
        print(f"\nError saving file: {e}")

    plt.show()

if __name__ == "__main__":
    visualize_heatmap()

