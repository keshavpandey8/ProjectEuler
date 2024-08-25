# Keshav Pandey
import math
from pathlib import Path

# Helper function that determines if input parameter is prime
# This prime number checker uses trial division algorithm
def check_prime(val: int) -> bool:
    # Smallest and only even prime number is 2
    if (val < 2):
        return False
    elif (val == 2):
        return True
    elif ((val % 2) == 0):
        return False

    # Input val is a positive odd number
    # Check if val has any factors. If so, then it is not prime
    max_prime = math.floor(math.sqrt(val))
    for i in range(3, max_prime+1, 2):
        if ((val % i) == 0):
            return False

    # No factors found for val. Therefore it must be prime
    return True


def ProjectEuler_AmicableNumbers_21() -> int:
    # Find sum of all amicable numbers less than n
    n = 10000
    sum = 0

    # Store (key:value) pairs of (value:sum of value's divisors) in below dict
    value_divisorsum = dict()

    for i in range(1, n):
        sum_of_divisors = 0
        for j in range(1, i):
            if ((i % j) == 0):
                sum_of_divisors += j
        value_divisorsum[i] = sum_of_divisors

        if (value_divisorsum.get(sum_of_divisors, -1) == i) and (i != sum_of_divisors):
            sum += (i + sum_of_divisors)

    return sum


def ProjectEuler_NamesScores_22() -> int:
    try:
        input_file = open(Path("input_files/0022_names.txt"), "r")
        names = input_file.read().replace('"', "").split(",")
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find list of first names")
        return -1

    # Sort list of names in alphabetical order
    names.sort()
    n = len(names)

    # Alphabetical score offset so that "A"=1, "B"=2, ..., "Z"=26
    offset = 64
    total_score = 0

    for i in range(n):
        curr_name = names[i]
        alpha_sum = 0

        for letter in curr_name:
            alpha_sum += (ord(letter) - offset)
        alpha_score = (i+1) * alpha_sum
        total_score += alpha_score

    return total_score


def ProjectEuler_NonAbundantSums_23() -> int:
    n = 28123
    abundant_nums = list()

    # Get list of all abundant numbers
    for i in range(2, n+1):
        divisor_sum = 1
        for curr_divisor in range(2, i):
            if ((i % curr_divisor) == 0):
                divisor_sum += curr_divisor
                if (divisor_sum > i):
                    break

        if (divisor_sum > i):
            abundant_nums.append(i)

    
    abundant_nums_set = set(abundant_nums)
    nonabundant_sum = 0

    for i in range(n+1):
        for abundant_num in abundant_nums:
            complement = i - abundant_num
            if (complement in abundant_nums_set):
                break
        else:
            nonabundant_sum += i

    return nonabundant_sum

# TODO: clean-up and understand modulo mathematics
def ProjectEuler_LexicographicPermutations_24() -> str:
    # List all digits being utilized in permutations from smallest to largest
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Find nth permutation of above digits
    n = 1000000

    # Check to make sure there are no repeated numbers in "digits"
    if (len(set(digits)) != len(digits)):
        print(f"Invalid list of digits: {digits}")
        return -1

    # Check to make sure 'digits' can have at least 'n' permutations
    if (n > math.factorial(len(digits))) or (n < 1):
        print(f"Invalid value of n: {n}")
        return -1

    result = ""
    curr_n = n
    while (len(digits) > 0):
        num_digits = len(digits)

        combos = math.factorial(num_digits-1)
        digit_idx = math.ceil(curr_n / combos) - 1
        curr_n = n % combos

        result += str(digits.pop(digit_idx))

    return result


def ProjectEuler_1000DigitFibonacci_25() -> int:
    # Initialize first 2 numbers in Fibonacci sequence
    fib1 = None
    fib2 = 1
    fib3 = 1

    # Start calculating values from index 2 until find value with n digits
    index = 2
    n = 1000

    while (len(str(fib3)) < n):
        fib1, fib2 = fib2, fib3
        fib3 = fib1 + fib2
        index += 1

    return index


# TODO: understand...(try by yourself and read comments as well)
def ProjectEuler_ReciprocalCycles_26() -> int:
    # Find longest recurring cycle in unit fraction with denominator less than 'n'
    n = 1000
    max_cycle_length = 0
    best_denom = 0

    # Start from 3 as '1' and '2' are both terminating fractions
    for i in range(1, n):
        curr_denom = i

        # Check if current unit_fraction is a terminating or repeating fraction
        while ((curr_denom % 2) == 0):
            curr_denom /= 2
        while ((curr_denom % 5) == 0):
            curr_denom /= 5
        if (curr_denom == 1):
            continue

        # Find length of recurring cycle for repeating fraction
        modval = 10 % curr_denom
        currmod = modval
        k = 1

        while (True):
            if (currmod == 1):
                break
            currmod = (currmod * modval) % curr_denom
            k += 1

        if (k > max_cycle_length):
            max_cycle_length = k
            best_denom = i

    return best_denom


def ProjectEuler_QuadraticPrimes_27() -> int:
    max_num_primes = 0
    product_of_coefficients = 0

    # Given quadratic equation: n^2 + an + b
    # Must have: |a| < 1000 and |b| <= 1000
    a_lim = 999
    b_lim = 1000

    for a in range(-a_lim, a_lim+1):
        for b in range(-b_lim, b_lim+1):
            n = 0
            soln = (math.pow(n, 2)) + (a * n) + (b)

            while (check_prime(soln)):
                n += 1
                soln = (math.pow(n, 2)) + (a * n) + (b)
            
            if (n > max_num_primes):
                max_num_primes = n
                product_of_coefficients = a * b

    return product_of_coefficients


def ProjectEuler_NumberSpiralDiagonals_28() -> int:
    # Find sum of diagonals in nxn spiral
    n = 1001
    spiral_val = 1
    diagonal_sum = 1

    for curr_spiral_len in range(1, n, 2):
        for _ in range(4):
            spiral_val += curr_spiral_len + 1
            diagonal_sum += spiral_val

    return diagonal_sum

    
def ProjectEuler_DistinctPowers_29() -> int:
    # Store all powers in a set
    distinct_powers = set()

    # Set min/max limits for 'a' and 'b'
    min_a = 2
    max_a = 100
    min_b = 2
    max_b = 100

    # Brute force: solve every combination of a^b and add solution to set
    for a_val in range(min_a, max_a+1):
        for b_val in range(min_b, max_b+1):
            curr_pow = math.pow(a_val, b_val)
            distinct_powers.add(curr_pow)

    return len(distinct_powers)


def ProjectEuler_DigitFifthPowers_30() -> int:
    n = 200000
    fifth_power_sum = 0

    for i in range(10, n+1):
        curr_val = i
        curr_sum = 0

        while (curr_val > 0):
            curr_digit = curr_val % 10
            curr_val = curr_val // 10
            curr_sum += math.pow(curr_digit, 5)

        if (curr_sum == i):
            fifth_power_sum += i

    return fifth_power_sum


if __name__ == "__main__":
    sol_21 = ProjectEuler_AmicableNumbers_21()
    print(f"sol_21 = {sol_21}")

    sol_22 = ProjectEuler_NamesScores_22()
    print(f"sol_22 = {sol_22}")

    sol_23 = ProjectEuler_NonAbundantSums_23()
    print(f"sol_23 = {sol_23}")

    sol_24 = ProjectEuler_LexicographicPermutations_24()
    print(f"sol_24 = {sol_24}")

    sol_25 = ProjectEuler_1000DigitFibonacci_25()
    print(f"sol_25 = {sol_25}")

    sol_26 = ProjectEuler_ReciprocalCycles_26()
    print(f"sol_26 = {sol_26}")

    sol_27 = ProjectEuler_QuadraticPrimes_27()
    print(f"sol_27 = {sol_27}")

    sol_28 = ProjectEuler_NumberSpiralDiagonals_28()
    print(f"sol_28 = {sol_28}")

    sol_29 = ProjectEuler_DistinctPowers_29()
    print(f"sol_29 = {sol_29}")

    sol_30 = ProjectEuler_DigitFifthPowers_30()
    print(f"sol_30 = {sol_30}")
