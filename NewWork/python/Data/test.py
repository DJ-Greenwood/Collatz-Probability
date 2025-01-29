import torch
import gmpy2
from gmpy2 import mpz
import os
import openpyxl
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class CollatzAnalyzerGPU:
    def __init__(self, chunk_size=1000, use_gpu=True):
        self.data = []  
        self.chunk_size = chunk_size
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    def collatz_sequence(self, n):
        """Generate the Collatz sequence with step tracking."""
        n = mpz(n)
        sequence = [n]
        max_value = n
        while n != 1:
            if n % 2 == 0:
                n //= 2
            else:
                n = 3 * n + 1
            sequence.append(n)
            max_value = max(max_value, n)
        return sequence, max_value

    def is_power_of_2(self, n):
        """Check if a number is a power of 2."""
        return n > 0 and (n & (n - 1)) == 0

    def analyze_collatz_phases(self, numbers):
        """Analyze Collatz sequences and track key properties."""
        results = []
        transition_counts = {'even_to_even': 0, 'even_to_odd': 0, 'odd_to_even': 0}
        convergence_times = []
        max_growth_ratios = []

        for n in numbers:
            n = mpz(n)
            sequence, max_value = self.collatz_sequence(n)
            first_power_of_2 = next((x for x in sequence if self.is_power_of_2(x)), None)

            # Compute Growth Ratio
            max_growth_ratio = max_value / n
            max_growth_ratios.append(max_growth_ratio)

            # Track Transitions
            for i in range(len(sequence) - 1):
                if sequence[i] % 2 == 0:
                    if sequence[i+1] % 2 == 0:
                        transition_counts['even_to_even'] += 1
                    else:
                        transition_counts['even_to_odd'] += 1
                else:
                    transition_counts['odd_to_even'] += 1

            convergence_times.append(len(sequence) - 1)

            result = {
                "number": str(n),
                "first_power_of_2": str(first_power_of_2) if first_power_of_2 else None,
                "sequence_length": len(sequence),
                "max_growth_ratio": max_growth_ratio,
                "total_steps": len(sequence) - 1
            }
            results.append(result)

        return results, transition_counts, convergence_times, max_growth_ratios


    def compute_statistics(self, convergence_times, transition_counts, max_growth_ratios):
        """Compute statistical properties."""
        
        # Convert gmpy2.mpfr objects to Python floats
        max_growth_ratios = [float(val) for val in max_growth_ratios]
        convergence_times = [float(val) for val in convergence_times]

        total_transitions = sum(transition_counts.values())
        transition_probs = {k: v / total_transitions for k, v in transition_counts.items()}

        mean_growth = np.mean(max_growth_ratios)
        std_growth = np.std(max_growth_ratios)  # Previously caused an error
        mean_convergence_time = np.mean(convergence_times)
        std_convergence_time = np.std(convergence_times)
        var_convergence_time = np.var(convergence_times)

        # Fit to a Negative Binomial Distribution
        r_est = (mean_convergence_time**2) / (var_convergence_time - mean_convergence_time)
        p_est = mean_convergence_time / var_convergence_time

        return transition_probs, mean_convergence_time, std_convergence_time, mean_growth, std_growth, r_est, p_est

    def plot_results(self, transition_probs, convergence_times, max_growth_ratios, r_est, p_est):
        """Plot key findings."""
        import matplotlib.pyplot as plt
        import numpy as np

        # Ensure all elements in max_growth_ratios are floats
        max_growth_ratios = np.array([float(val) for val in max_growth_ratios], dtype=np.float64)
        convergence_times = np.array([float(val) for val in convergence_times], dtype=np.float64)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Transition Probabilities
        axes[0].bar(transition_probs.keys(), transition_probs.values(), color=['blue', 'green', 'red'])
        axes[0].set_xlabel("Transition Type")
        axes[0].set_ylabel("Probability")
        axes[0].set_title("Collatz Transition Probabilities")

        # Convergence Time Distribution
        axes[1].hist(convergence_times, bins=30, density=True, alpha=0.6, label="Empirical Data")
        x = np.arange(min(convergence_times), max(convergence_times))
        axes[1].plot(x, stats.nbinom.pmf(x, r_est, p_est), 'r-', label="Fitted Negative Binomial")
        axes[1].set_xlabel("Convergence Time (Steps)")
        axes[1].set_ylabel("Probability Density")
        axes[1].set_title("Convergence Time Distribution")
        axes[1].legend()

        # Growth Ratios Histogram (Fixing the error)
        axes[2].hist(max_growth_ratios, bins=30, density=True, alpha=0.6, color='purple')
        axes[2].set_xlabel("Max Growth Ratio")
        axes[2].set_ylabel("Density")
        axes[2].set_title("Max Growth Ratio Distribution")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    analyzer = CollatzAnalyzerGPU(chunk_size=500, use_gpu=True)
    start = mpz(2)**1 + 1
    end = mpz(2)**2 + 1
    numbers = [start + i for i in range(1_000_000)]  # Smaller range for efficiency

    results, transition_counts, convergence_times, max_growth_ratios = analyzer.analyze_collatz_phases(numbers)
    transition_probs, mean_convergence_time, std_convergence_time, mean_growth, std_growth, r_est, p_est = analyzer.compute_statistics(convergence_times, transition_counts, max_growth_ratios)

    print("Transition Probabilities:", transition_probs)
    print("Mean Convergence Time:", mean_convergence_time)
    print("Standard Deviation of Convergence Time:", std_convergence_time)
    print("Mean Growth Ratio:", mean_growth)
    print("Standard Deviation of Growth Ratio:", std_growth)
    print("Estimated Negative Binomial Parameters (r, p):", (r_est, p_est))

    analyzer.plot_results(transition_probs, convergence_times, max_growth_ratios, r_est, p_est)
