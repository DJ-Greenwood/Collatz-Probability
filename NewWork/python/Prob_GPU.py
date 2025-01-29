import torch
import gmpy2
from gmpy2 import mpz
import os
import openpyxl

class CollatzAnalyzerGPU:
    def __init__(self, chunk_size=1000, use_gpu=True):
        self.data = []  # Stores analysis results for all numbers evaluated
        self.chunk_size = chunk_size
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    def collatz_sequence(self, n):
        """Generate the Collatz sequence starting from n using arbitrary precision."""
        n = mpz(n)  # Convert to arbitrary-precision integer
        sequence = [n]

        while n != 1:
            if n % 2 == 0:
                n //= 2
            else:
                n = 3 * n + 1
            sequence.append(n)

        return sequence

    def is_power_of_2(self, n):
        """Check if a number is a power of 2 (arbitrary precision)."""
        return n > 0 and (n & (n - 1)) == 0

    def analyze_collatz_phases(self, numbers):
        """Analyze Collatz sequences in chunks using GPU."""
        results = []
        
        for n in numbers:
            n = mpz(n)  # Convert to arbitrary precision
            sequence = self.collatz_sequence(n)
            initial_phase = []
            power_of_2_phase = []
            cycle_phase = []

            first_power_of_2 = next((x for x in sequence if self.is_power_of_2(x)), None)

            for num in sequence:
                initial_phase.append(num)
                if num == first_power_of_2:
                    break

            if first_power_of_2:
                power_of_2_phase_start = sequence.index(first_power_of_2) + 1
                power_of_2_phase_end = sequence.index(first_power_of_2) if 4 in sequence else len(sequence)
                power_of_2_phase = sequence[power_of_2_phase_start:power_of_2_phase_end]

            cycle_start = sequence.index(first_power_of_2) if 4 in sequence else len(sequence)
            cycle_phase = sequence[cycle_start:]

            result = {
                "number": str(n),  # Convert mpz to string for safe storage
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
        return results

    def save_data(self, filename):
        """Save the analyzed data to a file."""
        dir_url = "Python/Data/Collatz_Conjecture/"
        if not os.path.exists(dir_url):
            os.makedirs(dir_url)
        file_path = os.path.join(dir_url, f"{filename}.txt")
        with open(file_path, 'w') as file:
            for item in self.data:
                file.write(f"{item}\n")

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

# Example usage
if __name__ == "__main__":
    analyzer = CollatzAnalyzerGPU(chunk_size=500, use_gpu=True)

    # Define large number range using gmpy2
    start = mpz(2) ** 2529+1
    end = mpz(2) ** 2530 # Adjust the range for testing

    # Generate numbers safely in chunks using arbitrary precision
    numbers = [start + i for i in range(100)]  # Process only a subset for efficiency

    # Process in chunks
    for i in range(0, len(numbers), analyzer.chunk_size):
        chunk = numbers[i:i + analyzer.chunk_size]
        results = analyzer.analyze_collatz_phases(chunk)
        print(f"Processed chunk {i // analyzer.chunk_size + 1}: {results[0]}")  # Print first result of each chunk

    # Save the results
    # analyzer.save_data("collatz_analysis_gpu")
    analyzer.save_to_excel("collatz_analysis_gpu_excel_2529")