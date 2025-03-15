import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import cauchy, poisson, uniform
import os

sns.set(style="whitegrid")
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def generate_samples(dist_name, size):
    if dist_name == "normal":
        return np.random.normal(loc=0, scale=1, size=size)
    elif dist_name == "cauchy":
        return cauchy.rvs(loc=0, scale=1, size=size)
    elif dist_name == "poisson":
        return np.random.poisson(lam=10, size=size)
    elif dist_name == "uniform":
        return uniform.rvs(loc=-np.sqrt(3), scale=2*np.sqrt(3), size=size)

def tukey_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((data < lower) | (data > upper)).sum()
    return outliers

def plot_boxplot(data, dist_label, size):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data, orient="h", color="skyblue", fliersize=5)
    plt.title(f"Boxplot: {dist_label}, size={size}")
    plt.xlabel("Value")
    filename = f"{output_dir}/boxplot_{dist_label}_{size}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    distributions = {
        "normal": "Нормальное",
        "cauchy": "Коши",
        "poisson": "Пуассона",
        "uniform": "Равномерное"
    }
    sizes = [20, 100, 1000]
    outlier_counts = {size: {dist: 0 for dist in distributions.values()} for size in sizes}

    for dist_key, dist_label in distributions.items():
        for size in sizes:
            sample = generate_samples(dist_key, size)
            outlier_count = tukey_outliers(sample)
            outlier_counts[size][dist_label] = outlier_count
            plot_boxplot(sample, dist_key, size)

    table_data = []
    for size in sizes:
        row = [size] + [outlier_counts[size][dist_label] for dist_label in distributions.values()]
        table_data.append(row)

    df = pd.DataFrame(table_data, columns=["Размер выборки"] + list(distributions.values()))
    print(df)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.scale(1, 2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_table.png")
    plt.close()

if __name__ == "__main__":
    main()
