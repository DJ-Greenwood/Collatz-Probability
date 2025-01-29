class CollatzAnalyzer:
    def __init__(self):
        self.data = []  # Stores analysis results for all numbers evaluated

    def collatz_sequence(self, n):
        """Generate the Collatz sequence starting from n, stopping dynamically at the funnel."""
        sequence = [n]
        power_of_2_steps = []  # Track steps that are powers of 2

        while n != 1:
            if self.is_power_of_2(n):
                power_of_2_steps.append(n)  # Record power-of-2 steps
            if n % 2 == 0:
                n //= 2
            else:
                n = 3 * n + 1
            sequence.append(n)

            # Dynamically determine if we're in a repeating funnel
            if self.is_in_funnel(n):
                power_of_2_steps.append(int(n/2))  # Record power-of-2 steps
                break

        return sequence, power_of_2_steps, n  # Include where it stopped

    def is_power_of_2(self, n):
        """Check if a number is a power of 2."""
        return n > 0 and (n & (n - 1)) == 0

    def is_in_funnel(self, n):
        """Check if a number is in a known funnel (e.g., [4, 2, 1] or similar cycles)."""
        number = n
        if self.is_power_of_2(number):
            return n 
        # Add more known funnels if needed
        return False

    def analyze_collatz_phases(self, n):
        """Analyze the phases of the Collatz sequence."""
        sequence, power_of_2_steps, stopping_point = self.collatz_sequence(n)
        initial_phase = []

        # Identify the initial phase up to the stopping point
        for num in sequence:
            initial_phase.append(num)
            if num == stopping_point:  # Stop at the dynamic stopping point
                break

        result = {
            "number": n,
            "first_power_of_2": next((x for x in sequence if self.is_power_of_2(x)), None),
            "stopping_point": stopping_point,
            "sequence": sequence,
            "power_of_2_steps": int(stopping_point/2),
            "initial_phase": initial_phase,
            "steps": {
                "initial_phase": len(initial_phase),  # Steps in the initial phase
                "power_of_2_steps": int(stopping_point/2),  # Count of power-of-2 steps
                "remaining_steps": len(sequence) - len(initial_phase),  # Steps after stopping
                "total": len(sequence) + int(stopping_point/2) - 4  # Total steps to stopping point
            }
        }
        self.data.append(result)
        return result

# Example usage
if __name__ == "__main__":
    analyzer = CollatzAnalyzer()

    # Test for specific numbers
    numbers = [5, 7, 31, 81, 120_000_345_678_910_345, 1_000_000_000_000_000_090_000_000_000_000_000_000_000_000]
    # for num in range(1, 1_000): # Uncomment to test a range of numbers
    for num in numbers: # Test for specific numbers
        phases = analyzer.analyze_collatz_phases(num)
        sequence = phases["sequence"]
        print(f"Number: {num}")
        print(f"Full Collatz sequence: {sequence}")
        print(f"Stopping Point: {phases['stopping_point']}")
        print(f"Power-of-2 steps: {phases['power_of_2_steps']}")
        print(f"Phases: {phases['steps']}")
        print("-" * 50)
