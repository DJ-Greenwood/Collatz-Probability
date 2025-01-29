import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

# Function to generate Collatz sequences and track transitions
def collatz_sequence(n):
    sequence = []
    transitions = {'even_to_even': 0, 'even_to_odd': 0, 'odd_to_even': 0}
    while n != 1:
        sequence.append(n)
        if n % 2 == 0:
            if (n // 2) % 2 == 0:
                transitions['even_to_even'] += 1
            else:
                transitions['even_to_odd'] += 1
            n = n // 2
        else:
            transitions['odd_to_even'] += 1
            n = 3 * n + 1
    sequence.append(1)
    return sequence, transitions, len(sequence) - 1

# Run simulations for multiple starting values
num_samples = 100000
results = []
transition_counts = {'even_to_even': 0, 'even_to_odd': 0, 'odd_to_even': 0}

for n in range(1, num_samples + 1):
    _, transitions, steps = collatz_sequence(n)
    results.append(steps)
    for key in transitions:
        transition_counts[key] += transitions[key]

# Compute transition probabilities
total_transitions = sum(transition_counts.values())
transition_probs = {k: v / total_transitions for k, v in transition_counts.items()}

# Compute empirical distribution of convergence times
convergence_times = np.array(results)
mean_convergence_time = np.mean(convergence_times)
std_convergence_time = np.std(convergence_times)

# Estimating parameters for the negative binomial distribution
var_convergence_time = np.var(convergence_times)
r_est = (mean_convergence_time**2) / (var_convergence_time - mean_convergence_time)
p_est = mean_convergence_time / var_convergence_time

# Visualize results
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(convergence_times, bins=100, density=True, alpha=0.6, label="Empirical Data")
x = np.arange(min(convergence_times), max(convergence_times))
ax.plot(x, stats.nbinom.pmf(x, r_est, p_est), 'r-', label="Fitted Negative Binomial")
ax.set_xlabel("Convergence Time (Steps)")
ax.set_ylabel("Probability Density")
ax.set_title("Empirical vs. Theoretical Distribution of Collatz Convergence Times")
ax.legend()
plt.show()

# Display transition probabilities and convergence statistics
print("Transition Probabilities:", transition_probs)
print("Mean Convergence Time:", mean_convergence_time)
print("Standard Deviation of Convergence Time:", std_convergence_time)
print("Estimated Negative Binomial Parameters (r, p):", (r_est, p_est))