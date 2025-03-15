import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, poisson, uniform
import gc


def round_physics_style(mean, variance):
    std_dev = np.sqrt(variance)
    if std_dev == 0:
        return round(mean, 3), round(variance, 3)

    precision = max(-int(np.floor(np.log10(std_dev))), 0)
    precision = max(precision, 3)

    # Возвращаем округленные значения
    return round(mean, precision), round(variance, precision)


def generate_samples(dist_name, size):
    if dist_name == "normal":
        return np.random.normal(loc=0, scale=1, size=size)
    elif dist_name == "cauchy":
        return np.random.standard_cauchy(size=size)
    elif dist_name == "poisson":
        return np.random.poisson(lam=10, size=size)
    elif dist_name == "uniform":
        return np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=size)


def plot_individual_distributions(distribution, sample_sizes):
    for size in sample_sizes:
        data = generate_samples(distribution, size)
        plt.figure(figsize=(10, 6))

        if size == 10:
            if distribution == 'poisson' or distribution == 'uniform' or distribution=='cauchy':
                bins = 3
            else:
                bins = 5
        elif size == 50:
            if distribution == 'poisson' or distribution == 'uniform':
                bins = 11
            else:
                bins = 15
        elif size == 1000:
            if distribution == 'poisson' or distribution == 'normal':
                bins = 15
            else:
                bins = 30
        plt.hist(data, bins=bins, alpha=0.6, density=True, label=f'n = {size}')
        x = np.linspace(min(data), max(data), 1000)

        if distribution == "normal":
            plt.plot(x, norm.pdf(x, 0, 1), 'r-', label='PDF N(0,1)')
        elif distribution == "cauchy":
            plt.plot(x, cauchy.pdf(x, 0, 1), 'r-', label='PDF C(0,1)')
        elif distribution == "poisson":
            values, counts = np.unique(data, return_counts=True)
            plt.plot(values, counts / size, 'ko', label='Poisson PMF')
        elif distribution == "uniform":
            plt.plot(x, uniform.pdf(x, loc=-np.sqrt(3), scale=2 * np.sqrt(3)), 'r-', label='PDF U(-√3, √3)')

        plt.title(f'{distribution.upper()} — n = {size}')
        plt.xlabel('Значения')
        plt.ylabel('Плотность')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{distribution}_n{size}.png')
        plt.close()
        gc.collect()

def plot_combined_distribution(distribution, sample_sizes):
    plt.figure(figsize=(12, 6))
    for size in sample_sizes:
        data = generate_samples(distribution, size)

        if size == 10:
            if distribution == 'poisson' or distribution == 'uniform':
                bins = 3
            else:
                bins = 5
        elif size == 50:
            if distribution == 'poisson' or distribution == 'normal':
                bins = 8
            else:
                bins = 15
        elif size == 1000:
            if distribution == 'poisson' or distribution == 'normal':
                bins = 15
            else:
                bins = 30
        plt.hist(data, bins=bins, alpha=0.5, density=True, label=f'n = {size}')
    x = np.linspace(-10, 10, 1000)

    if distribution == "normal":
        plt.plot(x, norm.pdf(x, 0, 1), 'r-', linewidth=2, label='PDF N(0,1)')
    elif distribution == "cauchy":
        plt.plot(x, cauchy.pdf(x, 0, 1), 'r-', linewidth=2, label='PDF C(0,1)')
    elif distribution == "poisson":
        plt.title(f'Общий график Poisson — только гистограммы (PMF отдельно)')
    elif distribution == "uniform":
        plt.plot(x, uniform.pdf(x, loc=-np.sqrt(3), scale=2 * np.sqrt(3)), 'r-', linewidth=2, label='PDF U(-√3, √3)')

    plt.title(f'Общий график для {distribution.upper()} распределения')
    plt.xlabel('Значения')
    plt.ylabel('Плотность')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{distribution}_combined.png')
    plt.close()
    gc.collect()


def compute_location_stats(data):
    mean = np.mean(data)
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    zq = (q1 + q3) / 2
    return mean, median, zq


def perform_experiment(sample_sizes, distributions, num_repeats=1000):
    results = []

    for dist in distributions:
        for size in sample_sizes:
            means, medians, zqs = [], [], []

            for _ in range(num_repeats):
                sample = generate_samples(dist, size)
                mean, median, zq = compute_location_stats(sample)
                means.append(mean)
                medians.append(median)
                zqs.append(zq)

            means = np.array(means)
            medians = np.array(medians)
            zqs = np.array(zqs)

            mean_rounded, mean_variance_rounded = round_physics_style(np.mean(means), np.var(means))
            median_rounded, median_variance_rounded = round_physics_style(np.mean(medians), np.var(medians))
            zq_rounded, zq_variance_rounded = round_physics_style(np.mean(zqs), np.var(zqs))

            results.append({
                'Распределение': dist,
                'Размер выборки': size,
                'E(среднее)': mean_rounded,
                'D(среднее)': mean_variance_rounded,
                'E(медиана)': median_rounded,
                'D(медиана)': median_variance_rounded,
                'E(zQ)': zq_rounded,
                'D(zQ)': zq_variance_rounded
            })

            del means, medians, zqs
            gc.collect()

    return pd.DataFrame(results)


def save_table_as_image(df, filename='statistics_table.png', title='Характеристики положения и дисперсии'):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import gc

    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['font.family'] = 'serif'

    df.columns = [
        'Распределение',
        'Размер выборки',
        r'$E(\bar{x})$',
        r'$D(\bar{x})$',
        r'$E(\mathrm{med}\ x)$',
        r'$D(\mathrm{med}\ x)$',
        r'$E(z_Q)$',
        r'$D(z_Q)$'
    ]

    fig, ax = plt.subplots(figsize=(18, len(df) * 0.6 + 1))
    ax.axis('off')
    plt.title(title, fontsize=18, pad=20, weight='bold')

    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    gc.collect()


if __name__ == "__main__":
    sample_sizes_hist = [10, 50, 1000]
    sample_sizes_stats = [10, 100, 1000]
    distributions = ['normal', 'cauchy', 'poisson', 'uniform']

    for dist in distributions:
        plot_combined_distribution(dist, sample_sizes_hist)
        plot_individual_distributions(dist, sample_sizes_hist)

    df_results = perform_experiment(sample_sizes_stats, distributions)
    save_table_as_image(df_results, filename="statistics_results_table.png")

    del df_results
    gc.collect()
