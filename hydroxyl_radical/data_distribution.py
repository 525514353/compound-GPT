from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def data_distribution():
    plt.rcParams["font.family"] = "Times new roman"
    data = pd.read_excel('data.xlsx')
    data_1 = data[data['outliers'] != 1]
    x = data_1.iloc[:, 4]
    plt.figure(figsize=(8, 6), dpi=300)

    # 绘制直方图
    plt.hist(x=x, bins=60, color='skyblue', edgecolor='black')

    # 添加平均值的竖线
    mean_value = x.mean()
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2)
    # 添加标注为平均值
    plt.text(mean_value, plt.ylim()[1] * 0.9, f'Mean: {mean_value:.2f}', color='red', fontsize=15, fontweight='bold',
             ha='right')

    # 设置标签和标题
    plt.xlabel('Value', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')
    plt.title('Distribution of RCH', fontsize=18, fontweight='bold')

    # 设置刻度字体大小和加粗
    plt.xticks(rotation=0, fontsize=14, fontweight='bold')
    plt.yticks(rotation=0, fontsize=14, fontweight='bold')

    # 显示网格
    plt.grid(True)

    # 调整布局并显示图形
    plt.tight_layout()
    plt.show()


data_distribution()
