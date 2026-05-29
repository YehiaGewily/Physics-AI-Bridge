
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ising_simulation.hopfield.core import HopfieldNetwork
from ising_simulation.hopfield.utils import create_letter_pattern


# ── Constants ──────────────────────────────────────────────
N = 100                     # neurons
LETTERS = ['N', 'O', 'V']  # stored patterns
N_TRIALS = 500
MAX_STEPS = 100
HAMMING_THRESHOLD = 10     # raw Hamming distance
CORRELATION_THRESHOLD = 0.6


# ── Helpers ────────────────────────────────────────────────
def hamming_distance(a, b):
    """Raw number of positions that differ."""
    return int(np.sum(a != b))


def correlation(a, b):
    """Normalised dot product:  corr ∈ [-1, 1]."""
    return float(np.dot(a, b)) / len(a)


def classify_state(state, stored_patterns):
    """
    Classify a converged state into one of four categories.
    Returns (category_str, hamming_to_nearest_stored).
    """
    # 1. Stored?
    for pat in stored_patterns:
        if hamming_distance(state, pat) < HAMMING_THRESHOLD:
            return "Stored", hamming_distance(state, pat)

    # 2. Inverted?
    for pat in stored_patterns:
        if hamming_distance(state, -pat) < HAMMING_THRESHOLD:
            return "Inverted", hamming_distance(state, pat)

    # 3. Mixture?  Check sign-thresholded mean of every pair (and its negation)
    n_p = len(stored_patterns)
    for i in range(n_p):
        for j in range(i + 1, n_p):
            pair_sum = stored_patterns[i].astype(float) + stored_patterns[j].astype(float)
            pair_mean = np.sign(pair_sum)
            pair_mean[pair_mean == 0] = 1

            if abs(correlation(state, pair_mean)) > CORRELATION_THRESHOLD:
                return "Mixture", hamming_distance(state, stored_patterns[i])

    # 4. Everything else
    nearest = min(hamming_distance(state, p) for p in stored_patterns)
    return "Unknown Attractor", nearest


def cluster_states(states, threshold=HAMMING_THRESHOLD):
    """
    Group states so that members within 'threshold' Hamming distance
    share a cluster.  Returns [(representative, count), ...] sorted
    by count descending.
    """
    clusters = []  # (representative, count)
    for s in states:
        matched = False
        for idx, (rep, cnt) in enumerate(clusters):
            if hamming_distance(s, rep) < threshold:
                clusters[idx] = (rep, cnt + 1)
                matched = True
                break
        if not matched:
            clusters.append((s.copy(), 1))
    clusters.sort(key=lambda x: x[1], reverse=True)
    return clusters


# ── Main experiment ────────────────────────────────────────
def run_experiment_2b():
    print("=" * 60)
    print("Experiment 2b — Spurious States Detection")
    print("=" * 60)

    L_grid = 10
    stored_patterns = [create_letter_pattern(l, L_grid) for l in LETTERS]
    net = HopfieldNetwork(n_neurons=N)
    net.train(stored_patterns)
    print(f"Trained on {LETTERS}  (N={N})")

    # Energies of stored patterns (reference)
    stored_energies_map = {}
    for letter, pat in zip(LETTERS, stored_patterns):
        stored_energies_map[letter] = net.compute_energy(pat)
    print("Stored-pattern energies:",
          ", ".join(f"{l}={e:.2f}" for l, e in stored_energies_map.items()))

    # ── Run trials ─────────────────────────────────────────
    classifications = []
    energies = []
    states = []
    hamming_to_stored = []

    print(f"\nRunning {N_TRIALS} trials …")
    for t in range(N_TRIALS):
        random_state = np.random.choice([-1, 1], size=N)
        net.state = random_state.copy()
        net.recall(random_state, max_steps=MAX_STEPS, mode='async')

        cat, h_dist = classify_state(net.state, stored_patterns)
        classifications.append(cat)
        energies.append(net.energy())
        states.append(net.state.copy())
        hamming_to_stored.append(h_dist)

    counter = Counter(classifications)
    print("Done.\n")

    # ── Derived statistics ─────────────────────────────────
    stored_e   = [e for c, e in zip(classifications, energies) if c == "Stored"]
    inverted_e = [e for c, e in zip(classifications, energies) if c == "Inverted"]
    mixture_e  = [e for c, e in zip(classifications, energies) if c == "Mixture"]
    unknown_e  = [e for c, e in zip(classifications, energies) if c == "Unknown Attractor"]

    legitimate_e = stored_e + inverted_e          # stored + inverted
    spurious_e   = mixture_e + unknown_e           # all spurious

    spurious_states = [s for c, s in zip(classifications, states)
                       if c in ("Mixture", "Unknown Attractor")]
    spurious_clusters = cluster_states(spurious_states) if spurious_states else []

    all_clusters = cluster_states(states)
    unique_spurious = [rep for rep, _ in spurious_clusters]

    mean_spurious_e = np.mean(spurious_e) if spurious_e else float('nan')
    sem_spurious_e  = (np.std(spurious_e, ddof=1) / np.sqrt(len(spurious_e))) if len(spurious_e) > 1 else 0.0
    mean_legitimate_e = np.mean(legitimate_e) if legitimate_e else float('nan')
    sem_legitimate_e  = (np.std(legitimate_e, ddof=1) / np.sqrt(len(legitimate_e))) if len(legitimate_e) > 1 else 0.0
    energy_gap = mean_spurious_e - mean_legitimate_e if not (np.isnan(mean_spurious_e) or np.isnan(mean_legitimate_e)) else float('nan')

    # Unweighted arithmetic mean of stored pattern energies (not visit-frequency-weighted)
    unweighted_stored_mean = np.mean(list(stored_energies_map.values()))

    # ── Terminal summary ───────────────────────────────────
    lines = []
    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * 60)
    p("RESULTS SUMMARY")
    p("=" * 60)
    p(f"Total trials:                 {N_TRIALS}")
    p(f"Convergence to stored:        {counter['Stored']:3d} ({counter['Stored']/N_TRIALS*100:5.1f}%)")
    p(f"Convergence to inverted:      {counter['Inverted']:3d} ({counter['Inverted']/N_TRIALS*100:5.1f}%)")
    p(f"Mixture states:               {counter['Mixture']:3d} ({counter['Mixture']/N_TRIALS*100:5.1f}%)")
    p(f"Unknown attractors:           {counter['Unknown Attractor']:3d} ({counter['Unknown Attractor']/N_TRIALS*100:5.1f}%)")
    p(f"Number of unique spurious attractors: {len(spurious_clusters)}")
    p()
    p("Energy of stored patterns:")
    for l, e in stored_energies_map.items():
        p(f"  {l}: E = {e:.2f}")
    p(f"Mean energy of spurious states:         {mean_spurious_e:.2f} (±SEM {sem_spurious_e:.2f})")
    p(f"Mean energy of stored+inverted (visit-weighted): {mean_legitimate_e:.2f} (±SEM {sem_legitimate_e:.2f})")
    p(f"Unweighted mean of stored pattern energies:      {unweighted_stored_mean:.2f}")
    p(f"Energy gap (spurious \u2212 legitimate):              {energy_gap:.2f}")
    p()

    if not spurious_clusters:
        p("WARNING: No spurious attractors found. Consider adjusting thresholds.")
    if counter["Mixture"] == 0:
        p("NOTE: Zero mixture states detected — correlation threshold may need tuning.")

    # ── Save CSV ───────────────────────────────────────────
    csv_path = "results/hopfield/exp2b_results.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("trial,classification,final_energy,hamming_to_nearest_stored\n")
        for t in range(N_TRIALS):
            f.write(f"{t},{classifications[t]},{energies[t]:.6f},{hamming_to_stored[t]}\n")
    print(f"CSV saved to {csv_path}")

    # ── Save summary txt ───────────────────────────────────
    txt_path = "results/hopfield/exp2b_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Summary saved to {txt_path}")

    # ── Figure ──────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))

    # Panel A — Pie chart
    ax_a = fig.add_subplot(2, 2, 1)
    cat_order   = ["Stored", "Inverted", "Mixture", "Unknown Attractor"]
    cat_colors  = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]
    pie_sizes   = [counter[c] for c in cat_order if counter[c] > 0]
    pie_labels  = [f"{c}\n{counter[c]} ({counter[c]/N_TRIALS*100:.1f}%)"
                  for c in cat_order if counter[c] > 0]
    pie_colors  = [col for c, col in zip(cat_order, cat_colors) if counter[c] > 0]
    ax_a.pie(pie_sizes, labels=pie_labels, colors=pie_colors,
             startangle=90, textprops={'fontsize': 9})
    ax_a.set_title("Final State Classification (500 random starts)",
                   fontsize=11, fontweight='bold')

    # Panel B — Energy histogram
    ax_b = fig.add_subplot(2, 2, 2)
    all_e = np.array(energies)
    bins = np.linspace(all_e.min() - 1, all_e.max() + 1, 35)
    if legitimate_e:
        ax_b.hist(legitimate_e, bins=bins, alpha=0.6, color='#2ecc71',
                  edgecolor='black', label='Stored + Inverted')
    if spurious_e:
        ax_b.hist(spurious_e, bins=bins, alpha=0.6, color='#e74c3c',
                  edgecolor='black', label='Spurious (Mixture + Unknown)')
    ref_energy = stored_energies_map[LETTERS[0]]
    ax_b.axvline(x=ref_energy, color='black', linestyle='--', linewidth=1.2,
                 label=f'Stored-pattern energy ({ref_energy:.1f})')
    ax_b.set_title("Energy Distribution: Stored vs Spurious States",
                   fontsize=11, fontweight='bold')
    ax_b.set_xlabel("Energy")
    ax_b.set_ylabel("Count")
    ax_b.legend(fontsize=8)

    # Panel C — Gallery of top 6 spurious attractors (full bottom row)
    n_show = min(6, len(spurious_clusters))
    if n_show > 0:
        for i in range(n_show):
            ax_c = fig.add_subplot(2, 6, 7 + i)
            rep, count = spurious_clusters[i]
            e = net.compute_energy(rep)
            ax_c.imshow(rep.reshape(10, 10), cmap='binary', vmin=-1, vmax=1)
            ax_c.set_title(f"Spurious #{i+1}\nE = {e:.2f}\nn = {count}",
                           fontsize=8)
            ax_c.axis('off')
        # hide unused slots
        for i in range(n_show, 6):
            ax_blank = fig.add_subplot(2, 6, 7 + i)
            ax_blank.axis('off')
    else:
        ax_blank = fig.add_subplot(2, 1, 2)
        ax_blank.text(0.5, 0.5, 'No spurious attractors found',
                      ha='center', va='center', fontsize=13)
        ax_blank.axis('off')

    fig.suptitle("Most Common Spurious Attractor States",
                 fontsize=14, fontweight='bold')
    fig.subplots_adjust(hspace=0.50, wspace=0.35, top=0.92)

    out_path = "results/hopfield/exp2b_spurious_states.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    run_experiment_2b()
