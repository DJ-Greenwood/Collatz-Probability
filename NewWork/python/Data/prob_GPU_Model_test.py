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
        self.data = []  # Stores analysis results for all numbers evaluated
        self.chunk_size = chunk_size
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    def collatz_sequence(self, n):
        """Generate the Collatz sequence starting from n using arbitrary precision."""
        n = mpz(n)
        sequence = [n]
        while n != 1:
            if n % 2 == 0:
                n //= 2
            else:
                n = 3 * n + 1
            sequence.append(n)
        return sequence

    def is_power_of_2(self, n):
        """Check if a number is a power of 2."""
        return n > 0 and (n & (n - 1)) == 0

    def analyze_collatz_phases(self, numbers):
        """Analyze Collatz sequences in chunks using GPU."""
        results = []
        transition_counts = {'even_to_even': 0, 'even_to_odd': 0, 'odd_to_even': 0}
        convergence_times = []
        
        for n in numbers:
            n = mpz(n)
            sequence = self.collatz_sequence(n)
            initial_phase, power_of_2_phase, cycle_phase = [], [], []
            first_power_of_2 = next((x for x in sequence if self.is_power_of_2(x)), None)

            for num in sequence:
                initial_phase.append(num)
                if num == first_power_of_2:
                    break

            if first_power_of_2:
                power_of_2_phase_start = sequence.index(first_power_of_2) + 1
                power_of_2_phase = sequence[power_of_2_phase_start:]

            cycle_start = sequence.index(first_power_of_2) if first_power_of_2 else len(sequence)
            cycle_phase = sequence[cycle_start:]

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
                "initial_phase": [str(i) for i in initial_phase],
                "power_of_2_phase": [str(i) for i in power_of_2_phase],
                "cycle_phase": [str(i) for i in cycle_phase],
                "steps": {
                    "initial_phase": len(initial_phase),
                    "power_of_2_phase": len(power_of_2_phase),
                    "cycle_phase": len(cycle_phase),
                    "total": len(sequence) - 1
                }
            }
            results.append(result)
        
        self.data.extend(results)
        return results, transition_counts, convergence_times
    
    def compute_statistics(self, convergence_times, transition_counts):
        """Compute transition probabilities and convergence time statistics."""
        total_transitions = sum(transition_counts.values())
        transition_probs = {k: v / total_transitions for k, v in transition_counts.items()}
        mean_convergence_time = np.mean(convergence_times)
        std_convergence_time = np.std(convergence_times)
        var_convergence_time = np.var(convergence_times)
        r_est = (mean_convergence_time**2) / (var_convergence_time - mean_convergence_time)
        p_est = mean_convergence_time / var_convergence_time
        return transition_probs, mean_convergence_time, std_convergence_time, r_est, p_est
    
    def save_to_excel(self, filename):
        """Save the analyzed data to an Excel file."""
        dir_url = "Python/Data/Collatz_Conjecture/"
        if not os.path.exists(dir_url):
            os.makedirs(dir_url)
        file_path = os.path.join(dir_url, f"{filename}.xlsx")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Number", "First Power of 2", "Sequence Length", "Initial Phase", "Power of 2 Phase", "Cycle Phase", "Steps"])
        
        for item in self.data:
            ws.append([
                item["number"],
                item["first_power_of_2"],
                item["sequence_length"],
                ', '.join(item["initial_phase"]),
                ', '.join(item["power_of_2_phase"]),
                ', '.join(item["cycle_phase"]),
                str(item["steps"])
            ])
        
        wb.save(file_path)

if __name__ == "__main__":
    analyzer = CollatzAnalyzerGPU(chunk_size=500, use_gpu=True)
    start = mpz(2) ** 1+1
    end = mpz(2) ** 2+1
    numbers = [start + i for i in range(end)]
    
    results, transition_counts, convergence_times = analyzer.analyze_collatz_phases(numbers)
    transition_probs, mean_convergence_time, std_convergence_time, r_est, p_est = analyzer.compute_statistics(convergence_times, transition_counts)
    
    print("Transition Probabilities:", transition_probs)
    print("Mean Convergence Time:", mean_convergence_time)
    print("Standard Deviation of Convergence Time:", std_convergence_time)
    print("Estimated Negative Binomial Parameters (r, p):", (r_est, p_est))
    
    analyzer.save_to_excel("collatz_analysis_gpu_model_test")
