import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit_aer.primitives import SamplerV2
from basic_solver import MaxXorSat, solve
from qaoa_solver import solve_qaoa
from grover_solver import build_grover_circuit, determine_opti_k


def run_benchmark(entry: MaxXorSat, qaoa_reps=1, grover_iterations=None):
    """
    Exécute les algorithmes Exact, QAOA et Grover sur une instance donnée
    et retourne un dictionnaire de métriques.
    """
    results = {
        "n": entry.n,
        "m": entry.m,
        "exact_util": np.nan,
        "exact_time": np.nan,
        "qaoa_reps": qaoa_reps,
        "qaoa_util": np.nan,
        "qaoa_time": np.nan,
        "qaoa_depth": np.nan,
        "qaoa_gates": np.nan,
        "qaoa_ratio": np.nan,
        "grover_util": np.nan,
        "grover_time": np.nan,
        "grover_depth": np.nan,
        "grover_gates": np.nan,
        "grover_iters": np.nan,
        "grover_ratio": np.nan,
        "grover_qubits": np.nan,
    }

    # --- 1. Exact ---
    start_exact = time.time()
    try:
        sol_exact, util_exact = solve(entry)
        results["exact_time"] = time.time() - start_exact
        results["exact_util"] = util_exact
    except Exception as e:
        print(f"[Exact] Erreur: {e}")
        return results  # Impossible de continuer sans solution exacte de référence

    # --- 2. QAOA ---
    reps_qaoa = qaoa_reps
    start_qaoa = time.time()
    try:
        final_qc_qaoa, best_params, best_cost = solve_qaoa(entry, reps=reps_qaoa)

        # Mesure
        measured_qc_qaoa = final_qc_qaoa.copy()
        measured_qc_qaoa.measure_all()

        sampler = SamplerV2()
        job = sampler.run([measured_qc_qaoa])
        result = job.result()

        # Récupération des comptes de mesure
        counts = result[0].data.meas.get_counts()

        # Trouver l'état binaire le plus fréquent
        best_bitstring_qaoa = max(counts, key=counts.get)

        sol_qaoa = np.array([int(c) for c in (best_bitstring_qaoa)])

        res = np.dot(entry.A, sol_qaoa) % 2
        util_qaoa = np.sum(res == entry.b)

        results["qaoa_time"] = time.time() - start_qaoa
        results["qaoa_util"] = util_qaoa
        results["qaoa_depth"] = final_qc_qaoa.depth()
        results["qaoa_gates"] = sum(final_qc_qaoa.count_ops().values())
        results["qaoa_ratio"] = util_qaoa / util_exact if util_exact > 0 else 1.0

    except Exception as e:
        print(f"[QAOA] Erreur: {e}")

    # --- 3. Grover ---
    start_grover = time.time()
    try:
        k_grover = determine_opti_k(entry)

        if grover_iterations is None:
            iterations_grover = int(np.pi / 4 * np.sqrt(2**entry.m))
        else:
            iterations_grover = grover_iterations

        qc_grover = build_grover_circuit(
            entry, k=k_grover, iterations=iterations_grover
        )

        sampler = SamplerV2()
        job = sampler.run([qc_grover])
        result = job.result()

        counts = result[0].data.meas.get_counts()
        best_bitstring_grover = max(counts, key=counts.get)

        sol_grover = np.array([int(c) for c in (best_bitstring_grover)])
        # Inverser pour correspondre à l'ordre little-endian de Qiskit vers numpy
        sol_grover = sol_grover[::-1]

        if len(sol_grover) == entry.m:
            res = np.dot(entry.A, sol_grover) % 2
            util_grover = np.sum(res == entry.b)

            results["grover_time"] = time.time() - start_grover
            results["grover_util"] = util_grover
            results["grover_depth"] = qc_grover.depth()
            results["grover_gates"] = sum(qc_grover.count_ops().values())
            results["grover_iters"] = iterations_grover
            results["grover_qubits"] = qc_grover.num_qubits
            results["grover_ratio"] = (
                util_grover / util_exact if util_exact > 0 else 1.0
            )

    except Exception as e:
        print(f"[Grover] Erreur: {e}")

    return results


def generate_random_instance(n, m, seed=None):
    """
    Génère une instance aléatoire de MaxXorSat.
    """
    if seed is not None:
        np.random.seed(seed)
    A = np.random.randint(0, 2, size=(n, m)).tolist()
    b = np.random.randint(0, 2, size=n).tolist()
    return MaxXorSat(n, m, A, b)


def generate_random_instance_set(num_instances, n_range, m_range, seed=None):
    """
    Génère un ensemble d'instances aléatoires de MaxXorSat.
    """
    instances = []
    if seed is not None:
        np.random.seed(seed)

    for _ in range(num_instances):
        n = np.random.randint(n_range[0], n_range[1] + 1)
        m = np.random.randint(m_range[0], m_range[1] + 1)
        instances.append(generate_random_instance(n, m))

    return instances


def plot_results(df):
    """
    Génère des graphiques pour comparer les performances.
    """
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Comparaison QAOA vs Grover", fontsize=16)

    # 1. Temps d'exécution
    # On compare Exact, QAOA, Grover en temps
    df_time = df[["n", "exact_time", "qaoa_time", "grover_time"]].melt(
        "n", var_name="Algo", value_name="Temps (s)"
    )
    sns.lineplot(
        data=df_time, x="n", y="Temps (s)", hue="Algo", marker="o", ax=axes[0, 0]
    )
    axes[0, 0].set_title("Temps d'exécution moyen vs Nombre de clauses (n)")
    axes[0, 0].set_yscale("log")

    # 2. Qualité de la solution (Ratio par rapport à l'exact)
    df_quality = df[["n", "qaoa_ratio", "grover_ratio"]].melt(
        "n", var_name="Algo", value_name="Ratio Utilité"
    )
    sns.barplot(data=df_quality, x="n", y="Ratio Utilité", hue="Algo", ax=axes[0, 1])
    axes[0, 1].set_title("Qualité de la solution (Ratio / Exact)")
    axes[0, 1].set_ylim(0, 1.2)
    axes[0, 1].axhline(1.0, color="r", linestyle="--")

    # 3. Complexité : Profondeur
    df_depth = df[["n", "qaoa_depth", "grover_depth"]].melt(
        "n", var_name="Algo", value_name="Profondeur"
    )
    sns.lineplot(
        data=df_depth, x="n", y="Profondeur", hue="Algo", marker="o", ax=axes[1, 0]
    )
    axes[1, 0].set_title("Profondeur du circuit vs n")

    # 4. Complexité : Nombre de portes
    df_gates = df[["n", "qaoa_gates", "grover_gates"]].melt(
        "n", var_name="Algo", value_name="Nombre de portes"
    )
    sns.lineplot(
        data=df_gates,
        x="n",
        y="Nombre de portes",
        hue="Algo",
        marker="o",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Nombre de portes vs n")

    plt.tight_layout()
    plt.savefig("resultats_comparaison.png")
    print("Graphiques sauvegardés dans 'resultats_comparaison.png'")
    # plt.show() # Commenté pour éviter de bloquer si pas d'interface graphique


"""
===========================================
=================== Main ==================
===========================================
"""

if __name__ == "__main__":
    # Paramètres de l'évaluation
    print("Lancement de l'évaluation...")
    n_range = (3, 5)  # Nombre de clauses
    m_range = (3, 5)  # Nombre de variables
    num_instances = 10

    qaoa_reps_list = [1, 2]

    print(
        f"Génération de {num_instances} instances aléatoires (n={n_range}, m={m_range})..."
    )
    instances = generate_random_instance_set(num_instances, n_range, m_range, seed=42)

    all_results = []

    for rep in qaoa_reps_list:
        print(f"\n--- Évaluation avec QAOA reps = {rep} ---")
        for i, instance in enumerate(instances):
            print(
                f"Traitement instance {i + 1}/{num_instances} (n={instance.n}, m={instance.m})..."
            )
            res = run_benchmark(instance, qaoa_reps=rep)
            all_results.append(res)

    df = pd.DataFrame(all_results)



    print("\n=== MOYENNES ===")
    metric_cols = [c for c in df.columns if c not in ["n", "m"]]
    mean_metrics = df[metric_cols].mean(numeric_only=True)
    print(mean_metrics)

    # Génération des graphiques
    plot_results(df)
