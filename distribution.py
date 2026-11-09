import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 定义文件名 ---

# 我们要加载的原始分值热力图
RAW_SCORE_FILE = 'data/potential_score_raw.npy'
# 我们要保存的分布图文件名
OUTPUT_DISTRIBUTION_IMAGE = 'data/potential_distribution.png'


# --- 2. 主分析函数 ---

def analyze_distribution():
    print(f"开始分析潜力值分布，正在加载文件...")

    # 1. 检查文件是否存在
    if not os.path.exists(RAW_SCORE_FILE):
        print(f"错误：缺少文件 '{RAW_SCORE_FILE}'。")
        print("请先运行潜力分计算脚本。")
        return

    # 2. 加载数据
    try:
        score_raw = np.load(RAW_SCORE_FILE)
        print(f"成功加载原始分值热力图，形状: {score_raw.shape}")
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return

    # 3. 准备数据以供分析
    #    a. 将 2D 数组 "压平" 为 1D 数组
    scores_flat = score_raw.flatten()

    #    b. 关键步骤：只分析有意义的 (非零) 潜力值
    #       绝大多数网格的分数是0，会使直方图完全偏向0
    scores_nonzero = scores_flat[scores_flat > 1e-9]  # 使用一个小的阈值

    if len(scores_nonzero) == 0:
        print("错误：未找到任何非零的潜力值。无法进行分析。")
        return

    total_cells = scores_flat.size
    nonzero_cells = scores_nonzero.size
    percent_nonzero = (nonzero_cells / total_cells) * 100

    print("\n--- 潜力值统计分析 (仅限非零网格) ---")
    print(f"总网格数: {total_cells}")
    print(f"非零潜力网格数: {nonzero_cells} ({percent_nonzero:.2f}%)")
    print("-" * 30)
    print(f"均值 (Mean):   {np.mean(scores_nonzero):.4f}")
    print(f"标准差 (Std):  {np.std(scores_nonzero):.4f}")
    print(f"最小值 (Min):   {np.min(scores_nonzero):.4f}")
    print(f"中位数 (Median): {np.median(scores_nonzero):.4f} (50th percentile)")
    print("-" * 30)
    print(f"25th Percentile: {np.percentile(scores_nonzero, 25):.4f}")
    print(f"75th Percentile: {np.percentile(scores_nonzero, 75):.4f}")
    print(f"90th Percentile: {np.percentile(scores_nonzero, 90):.4f}")
    print(f"95th Percentile: {np.percentile(scores_nonzero, 95):.4f}")
    print(f"99th Percentile: {np.percentile(scores_nonzero, 99):.4f}")
    print(f"最大值 (Max):   {np.max(scores_nonzero):.4f}")
    print("------------------------------------------")

    # --- 4. 绘制直方图 ---
    print("\n正在生成潜力值分布直方图...")

    plt.figure(figsize=(12, 6))

    # 绘制直方图
    # bins='auto' 会自动选择合适的箱子数量
    # log=True: 将y轴设为对数刻度，这对于长尾分布非常重要
    plt.hist(scores_nonzero, bins='auto', log=True, edgecolor='black', alpha=0.7)

    plt.title('潜力值分布直方图 (仅非零值, Y轴对数刻度)', fontsize=16)
    plt.xlabel('原始潜力分值', fontsize=12)
    plt.ylabel('网格数量 (对数刻度)', fontsize=12)

    # 在图上添加中位数和95百分位的垂线
    median_val = np.median(scores_nonzero)
    p95_val = np.percentile(scores_nonzero, 95)

    plt.axvline(median_val, color='red', linestyle='dashed', linewidth=2)
    plt.axvline(p95_val, color='purple', linestyle='dashed', linewidth=2)
    plt.legend([
        f'中位数: {median_val:.2f}',
        f'95百分位: {p95_val:.2f}'
    ])

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 5. 保存图像
    plt.savefig(OUTPUT_DISTRIBUTION_IMAGE, dpi=150)

    print(f"\n--- 分析完成! ---")
    print(f"分布图已保存到: {OUTPUT_DISTRIBUTION_IMAGE}")
    plt.show()


if __name__ == "__main__":
    # 确保您安装了 matplotlib
    # pip install matplotlib
    analyze_distribution()