import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from matplotlib.patches import Ellipse
from tqdm import tqdm

# ================================
# НАСТРОЙКИ ЭКСПЕРИМЕНТА
# ================================
np.random.seed(42)
N_REPEATS = 1000
SAMPLE_SIZES = [20, 60, 100]
RHO_VALUES = [0, 0.5, 0.9]

# Смесь нормальных распределений
MIXTURE_WEIGHTS = [0.9, 0.1]
MIXTURE_PARAMS = [
    {"mean": [0, 0], "cov": [[1, 0.9], [0.9, 1]]},
    {"mean": [0, 0], "cov": [[10, -9], [-9, 10]]},
]


# ================================
# ФУНКЦИИ
# ================================

def generate_normal_sample(n, rho):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    return np.random.multivariate_normal(mean, cov, n)

def generate_mixture_sample(n):
    result = []
    for _ in range(n):
        component = np.random.choice([0, 1], p=MIXTURE_WEIGHTS)
        mean = MIXTURE_PARAMS[component]["mean"]
        cov = MIXTURE_PARAMS[component]["cov"]
        sample = np.random.multivariate_normal(mean, cov)
        result.append(sample)
    return np.array(result)

def quadrant_correlation(x, y):
    mx, my = np.median(x), np.median(y)
    q1 = np.logical_and(x >= mx, y >= my).sum()
    q2 = np.logical_and(x < mx, y >= my).sum()
    q3 = np.logical_and(x < mx, y < my).sum()
    q4 = np.logical_and(x >= mx, y < my).sum()
    return ((q1 + q3) - (q2 + q4)) / len(x)

def compute_correlations(x, y):
    rp = pearsonr(x, y)[0]
    rs = spearmanr(x, y)[0]
    rq = quadrant_correlation(x, y)
    return rp, rs, rq

def run_simulation(n, rho=None, is_mixture=False):
    results = []
    for _ in range(N_REPEATS):
        if is_mixture:
            sample = generate_mixture_sample(n)
        else:
            sample = generate_normal_sample(n, rho)
        x, y = sample[:, 0], sample[:, 1]
        results.append(compute_correlations(x, y))
    return np.array(results)

def describe_statistics(result_matrix):
    mean = result_matrix.mean(axis=0)
    mean_sq = (result_matrix**2).mean(axis=0)
    var = result_matrix.var(axis=0, ddof=1)
    return {
        "mean": mean,
        "mean_sq": mean_sq,
        "var": var
    }

def plot_sample_with_ellipse(sample, title, color='blue'):
    x, y = sample[:, 0], sample[:, 1]
    cov = np.cov(x, y)
    mean = np.mean(x), np.mean(y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=10, alpha=0.6)
    ell = Ellipse(xy=mean,
                  width=2 * np.sqrt(5.991) * lambda_[0],
                  height=2 * np.sqrt(5.991) * lambda_[1],
                  angle=np.rad2deg(np.arccos(v[0, 0])),
                  edgecolor=color, facecolor='none', lw=2)
    ax.add_patch(ell)
    ax.set_title(title)
    ax.grid(True)
    plt.axis('equal')
    plt.show()


# ================================
# ОСНОВНОЙ ЦИКЛ
# ================================

for n in SAMPLE_SIZES:
    for rho in RHO_VALUES:
        print(f"\n=== Классическое распределение: n={n}, rho={rho} ===")
        result = run_simulation(n, rho=rho)
        stats = describe_statistics(result)
        print(f"Средние:          Пирсон={stats['mean'][0]:.4f}, Спирмен={stats['mean'][1]:.4f}, Квадрант={stats['mean'][2]:.4f}")
        print(f"Средние квадраты: Пирсон={stats['mean_sq'][0]:.4f}, Спирмен={stats['mean_sq'][1]:.4f}, Квадрант={stats['mean_sq'][2]:.4f}")
        print(f"Дисперсии:        Пирсон={stats['var'][0]:.4f}, Спирмен={stats['var'][1]:.4f}, Квадрант={stats['var'][2]:.4f}")
        sample = generate_normal_sample(n, rho)
        plot_sample_with_ellipse(sample, f"Normal: n={n}, ρ={rho}", color='blue')

for n in SAMPLE_SIZES:
    print(f"\n=== Смесь нормальных распределений: n={n} ===")
    result = run_simulation(n, is_mixture=True)
    stats = describe_statistics(result)
    print(f"Средние:          Пирсон={stats['mean'][0]:.4f}, Спирмен={stats['mean'][1]:.4f}, Квадрант={stats['mean'][2]:.4f}")
    print(f"Средние квадраты: Пирсон={stats['mean_sq'][0]:.4f}, Спирмен={stats['mean_sq'][1]:.4f}, Квадрант={stats['mean_sq'][2]:.4f}")
    print(f"Дисперсии:        Пирсон={stats['var'][0]:.4f}, Спирмен={stats['var'][1]:.4f}, Квадрант={stats['var'][2]:.4f}")
    sample = generate_mixture_sample(n)
    plot_sample_with_ellipse(sample, f"Mixture: n={n}", color='red')
