import math
import os
from decimal import Decimal, getcontext

# Set precision for Decimal calculations
getcontext().prec = 1000

class CollatzAnalyzer:
    def __init__(self):
        self.data = []  # Stores analysis results for all numbers evaluated

    def collatz_sequence(self, n):
        """Generate the Collatz sequence starting from n."""
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

    def analyze_collatz_phases(self, n):
        """Analyze the phases of the Collatz sequence."""
        sequence = self.collatz_sequence(n)
        initial_phase = []
        power_of_2_phase = []
        cycle_phase = []

        first_power_of_2 = next((x for x in sequence if self.is_power_of_2(x)), None)
        print(f"First power of 2 in sequence: {first_power_of_2}")
        for num in sequence:
            initial_phase.append(num)
            if num == first_power_of_2:
                break

        if first_power_of_2:
            power_of_2_phase_start = sequence.index(first_power_of_2) + 1
            power_of_2_phase_end = sequence.index(4)
            power_of_2_phase = sequence[power_of_2_phase_start:power_of_2_phase_end]

        cycle_start = sequence.index(4)
        cycle_phase = sequence[cycle_start:]

        result = {
            "number": n,
            "first_power_of_2": first_power_of_2,
            "sequence": sequence,
            "initial_phase": initial_phase,
            "power_of_2_phase": power_of_2_phase,
            "cycle_phase": cycle_phase,
            "steps": {
                "initial_phase": len(initial_phase),
                "power_of_2_phase": len(power_of_2_phase),
                "cycle_phase": len(cycle_phase),
                "total": len(sequence) - 1
            }
        }
        self.data.append(result)
        return result

    def expected_steps_power_of_2(self, n):
        """Calculate expected steps for powers of 2."""
        return math.log2(n)

    def expected_steps_odd(self, n):
        """Calculate expected steps for odd numbers."""
        if n % 2 != 1:
            raise ValueError("Number must be odd.")
        return 2 + self.expected_steps(self.collatz_sequence(3 * n + 1)[1])

    def expected_steps_even_not_power_of_2(self, n):
        """Calculate expected steps for even numbers that are not powers of 2."""
        if self.is_power_of_2(n):
            raise ValueError("Number must not be a power of 2.")
        steps = 1 + self.expected_steps(self.collatz_sequence(n // 2)[1])
        return steps

    def expected_steps(self, n):
        """Calculate expected steps for any number."""
        if self.is_power_of_2(n):
            return self.expected_steps_power_of_2(n)
        elif n % 2 == 1:
            return self.expected_steps_odd(n)
        else:
            return self.expected_steps_even_not_power_of_2(n)

    def statistical_descent(self, sequence):
        """Analyze the statistical descent of a Collatz sequence."""
        even_steps = sum(1 for n in sequence if n % 2 == 0)
        odd_steps = len(sequence) - even_steps

        return {
            "total_steps": len(sequence) - 1,
            "even_steps": even_steps,
            "odd_steps": odd_steps,
            "even_step_reduction": float(even_steps * Decimal(0.5)),
            "odd_step_effect": float(odd_steps * Decimal(3) / Decimal(2))
        }

    def key_probabilities(self):
        """Calculate and display key probabilities based on the Collatz process."""
        # Probability of staying even in a single halving step
        p_even_to_even = 3 / 4

        # Probability of transitioning to a power of 2
        k = 1
        p_reach_power_of_2 = 1
        while True:
            term = (3 / 4) ** k
            if term < 1e-10:  # Convergence threshold
                break
            p_reach_power_of_2 *= term
            k += 1

        # Combined probability for odd-to-even transition and halving steps
        p_return_to_power_of_2 = sum([(1 / 4) ** k for k in range(1, 100)])  # Approximation
        p_return_to_power_of_2 = 1 / 3  # Simplified closed form

        # Expected time to return to a power of 2
        expected_steps = 1 / p_return_to_power_of_2

        # Results
        probabilities = {
            "P(even→even)": p_even_to_even,
            "P(reach power of 2)": p_reach_power_of_2,
            "P(return to power of 2)": p_return_to_power_of_2,
            "E[T(n)] (expected steps)": expected_steps
        }
        return probabilities

    def generate_report(self):
        """Generate a detailed summary report for all analyzed numbers."""
        total_numbers = len(self.data)
        total_steps = sum(item["steps"]["total"] for item in self.data)
        average_steps = float(Decimal(total_steps) / Decimal(total_numbers)) if total_numbers else 0
        max_steps = max(self.data, key=lambda x: x["steps"]["total"]) if self.data else None
        min_steps = min(self.data, key=lambda x: x["steps"]["total"]) if self.data else None

        report = {
            "total_numbers": total_numbers,
            "total_steps": total_steps,
            "average_steps": average_steps,
            "max_steps": {
                "number": max_steps["number"],
                "steps": max_steps["steps"]["total"]
            } if max_steps else None,
            "min_steps": {
                "number": min_steps["number"],
                "steps": min_steps["steps"]["total"]
            } if min_steps else None,
            "detailed_analysis": [
                {
                    "number": item["number"],
                    "total_steps": item["steps"]["total"],
                    "initial_phase_steps": item["steps"]["initial_phase"],
                    "power_of_2_phase_steps": item["steps"]["power_of_2_phase"],
                    "cycle_phase_steps": item["steps"]["cycle_phase"]
                } for item in self.data
            ]
        }
        return report

    def report_all_data(self):
        """Return all analyzed data."""
        return self.data
    
    def save_data(self, data, filename):
        """Save the analyzed data to a file."""
        dir_url = "Python/Data/Collatz_Conjecture/"
        if not os.path.exists(dir_url):
            os.makedirs(dir_url)
        file_path = os.path.join(dir_url, f"{filename}.txt")
        with open(file_path, 'w') as file:
            for item in data:
                file.write(f"{item}\n")

# Example usage
if __name__ == "__main__":
    analyzer = CollatzAnalyzer()

    for num in [7, 31, 27]: # Test for specific numbers
        phases = analyzer.analyze_collatz_phases(num)
        sequence = phases["sequence"]
        descent_analysis = analyzer.statistical_descent(sequence)

        print(f"Number: {num}")
        print(f"Full Collatz sequence: {sequence}")
        print(f"Phases: {phases['steps']}")
        print(f"Statistical descent: {descent_analysis}")
        print("-" * 50)

    summary = analyzer.generate_report()

    all_data = analyzer.report_all_data()
    save_filename = "collatz_analysis"
    analyzer.save_data(all_data, save_filename)