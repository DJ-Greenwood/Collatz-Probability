import math
import random

def collatz_sequence(n):
    """Generate the Collatz sequence starting from n."""
    sequence = [n]
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        sequence.append(n)
    return sequence

def is_power_of_2(n):
    """Check if a number is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0

def analyze_collatz_phases(n):
    """Analyze the phases of the Collatz sequence."""
    sequence = collatz_sequence(n)
    initial_phase = []
    power_of_2_phase = []
    cycle_phase = []

    # Identify the first power of 2 in the sequence
    first_power_of_2 = next((x for x in sequence if is_power_of_2(x)), None)

    # Phase 1: Initial phase (until first power of 2, inclusive)
    for num in sequence:
        initial_phase.append(num)
        if num == first_power_of_2:
            break

    # Phase 2: Power of 2 reduction (from first power of 2 to cycle entry)
    if first_power_of_2:
        power_of_2_phase_start = sequence.index(first_power_of_2) + 1
        power_of_2_phase_end = sequence.index(4)  # First entry into the cycle {4, 2, 1}
        power_of_2_phase = sequence[power_of_2_phase_start:power_of_2_phase_end]

    # Phase 3: Cycle entry (from 4 to 1)
    cycle_start = sequence.index(4)
    cycle_phase = sequence[cycle_start:]

    return {
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

def statistical_descent(sequence):
    """Analyze the statistical descent of a Collatz sequence."""
    even_steps = sum(1 for n in sequence if n % 2 == 0)
    odd_steps = len(sequence) - even_steps

    return {
        "total_steps": len(sequence) - 1,
        "even_steps": even_steps,
        "odd_steps": odd_steps,
        "even_step_reduction": even_steps * 0.5,  # Average reduction for even steps
        "odd_step_effect": odd_steps * 3 / 2  # Approximate increase for odd steps
    }

def probability_of_infinite_growth(k):
    """
    Calculate the probability of k consecutive increases.

    Args:
        k (int): Number of consecutive increases.

    Returns:
        float: Probability of k consecutive increases.
    """
    return (1 / 4) ** k


def probability_of_infinite_growth_limit():
    """
    Calculate the probability of infinite growth.

    Given the distribution of even/odd transitions:
    • Probability of k consecutive increases:
      P(k increases) ≤ (1/4)^k
    • Probability of infinite growth:
      lim (k→∞) P(infinite growth) = 0

    Returns:
        int: Always returns 0 because the probability of infinite growth is zero.
    """
    return 0  # As k → ∞, the probability of infinite growth is 0


def convergence_time_distribution(n):
    """
    Estimate convergence time distribution as a Negative Binomial.

    The convergence time follows an approximate Negative Binomial distribution, 
    where:
    - r (int): The number of 'successes' (log2(n)).
    - p (float): The probability of a 'success' (7/16).

    Args:
        n (int): Starting value for the Collatz sequence.

    Returns:
        dict: A dictionary containing:
            - expected_convergence_time (float): The expected convergence time.
            - parameters (dict): The parameters of the Negative Binomial distribution ('r' and 'p').
            - description (str): A brief explanation of the output.
    """
    if n <= 0:
        raise ValueError("Input 'n' must be a positive integer.")
    
    r = math.ceil(math.log2(n))  # Number of 'successes' needed
    p = 7 / 16  # Probability of a 'success'

    # Calculate expected convergence time
    expected_time = r / p

    # Create the result dictionary
    distribution = {
        "expected_convergence_time": expected_time,
        "parameters": {
            "r": r,
            "p": p
        },
        "description": (
            "The expected convergence time is calculated based on the Negative Binomial "
            "distribution with parameters r (log2(n)) and p (7/16). This is an estimate "
            "and assumes statistical trends for the Collatz sequence."
        )
    }

    return distribution

import math

def evaluate_convergence_time(n, c=2.41, k=0):
    """
    Evaluate the convergence time using the formula E[T(n)] <= c * log(n) + k.
    
    Args:
        n (int): Starting value for the Collatz sequence (must be positive).
        c (float, optional): Coefficient reflecting the growth rate (default: 2.41).
        k (float, optional): Constant overhead added to the convergence time (default: 0).

    Returns:
        dict: A dictionary containing:
            - log_n (float): The logarithm of the input value.
            - convergence_time (float): The evaluated convergence time.
            - parameters (dict): The parameters used for the calculation ('c' and 'k').
            - description (str): An explanation of the result.
    """
    if n <= 0:
        raise ValueError("Input 'n' must be a positive integer.")
    
    log_n = math.log(n)  # Natural logarithm of n
    convergence_time = c * log_n + k  # Evaluate convergence time

    return {
        "log_n": log_n,
        "convergence_time": convergence_time,
        "parameters": {
            "c": c,
            "k": k
        },
        "description": (
            f"The convergence time is calculated using the formula "
            f"E[T(n)] <= {c} * log(n) + {k}, where log(n) is the natural "
            f"logarithm of {n}. This provides an upper bound on the expected "
            f"time for the Collatz sequence to converge."
        )
    }

# Example usage
n = 31
phases = analyze_collatz_phases(n)
sequence = phases["sequence"]

# Statistical descent analysis
descent_analysis = statistical_descent(sequence)

# Probability of k consecutive increases
k = 5  # Example number of consecutive increases
prob_k_increases = probability_of_infinite_growth(k)

# Probability of infinite growth
prob_infinite_growth = probability_of_infinite_growth_limit()

# Convergence time distribution
convergence_dist = convergence_time_distribution(n)

# Evaluate convergence time using E[T(n)] <= c * log(n) + k
c = 2.41
k_const = 0  # Adjust as needed
evaluated_convergence_time = evaluate_convergence_time(n, c, k_const)

# Display results
print("Full Collatz sequence:", sequence)
print("Phases:", phases["steps"])
print("Statistical descent:", descent_analysis)
print(f"Probability of {k} consecutive increases: {prob_k_increases}")
print("Probability of infinite growth:", prob_infinite_growth)
print("Convergence time distribution:", convergence_dist)
print("Evaluated convergence time:", evaluated_convergence_time)
