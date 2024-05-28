from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
data_1=pd.read_csv('smiles_cas_N6512.csv',header=None,delimiter=' 	',engine='python')

def data_distribution():
    plt.rcParams["font.family"] = "Times new roman"

    plt.figure(figsize=(8, 6), dpi=300)

    counts = data_1.iloc[:, 2].value_counts()

    counts.plot(kind='bar', color='skyblue', edgecolor='black')

    plt.xlabel('Value', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')
    plt.title('Distribution of toxicity data', fontsize=18, fontweight='bold')

    plt.xticks(rotation=0, fontsize=14,fontweight='bold')
    plt.yticks(rotation=0, fontsize=14, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.savefig('Distribution of toxicity data.png', dpi=300)

    plt.show()
data_distribution()

