# Keshav Pandey
from collections import defaultdict
from itertools import permutations
import json
import math
from pathlib import Path

def get_next_continued_fraction(sqr_const: int, sqrt_const: int, num_val: int, denom_val: int) -> tuple[int, int, int]:
    # Solve for the next a_i constant
    next_a = math.floor(num_val / (sqrt_const-denom_val))

    # Solve for new denominator by multiplying by the conjugate
    new_denom = sqr_const - (denom_val*denom_val)

    # Check if we can reduce the denominator 'new_denom' constant with the numerator 'num_val' constant
    reduction_const = math.gcd(num_val, new_denom)
    num_val //= reduction_const
    new_denom //= reduction_const

    # If continued fraction pattern is followed correctly:
    # then the numerator constant multiplying the conjugate should always reduce to 1
    if (num_val != 1):
        print(f"Error: found non-zero numerator constant that cannot be reduced")
        return (-1, -1)

    # Solve for new numerator value by pulling out "next_a" constant from fraction
    new_num = (denom_val) - (new_denom*next_a)

    # Take reciprocal to get numerator & denominator vals for next continued fraction iteration:
    num_val = new_denom
    denom_val = -1 * new_num

    return (num_val, denom_val, next_a)


# a_vals starts from a_1 and goes to desired a_n
def get_convergent_fraction(a0: int, a_vals: list) -> tuple[int, int]:
    # Initialize numerator and denominator to last (most nested) fraction value
    numerator = 1
    denominator = a_vals[-1]

    # Solve all fractions from last to first
    for i in range(len(a_vals)-2, -1, -1):
        numerator += a_vals[i] * denominator
        numerator, denominator = denominator, numerator

    numerator += a0 * denominator

    return (numerator, denominator)


# Return concatenation of two positive integers (x+y) without any type casting
def concatenate_integers(x: int, y: int):
    order_y = 1
    while (order_y <= y):
        order_y *= 10

    return ((x * order_y) + y)


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


# Helper function that calculates phi and phi_ratio for val with only 2 prime factors
def get_phi_and_phi_ratio(val: int, p_factor1: int, p_factor2: int) -> tuple[int, int]:
    # Find phi(val) and val/phi(val)
    phi = (p_factor1 - 1) * (p_factor2 - 1)
    ratio = val / phi
    return (phi, ratio)


# Helper function that determines if num1 and num2 are permutations of each other
# Inputs num1, num2 must be > 0
def isPermutation(num1: int, num2: int) -> bool:
    digits = [0] * 10

    # Get count of each digit in num1
    while (num1 > 0):
        digits[num1 % 10] += 1
        num1 //= 10

    # Subtract count of each digit in num2
    while (num2 > 0):
        digits[num2 % 10] -= 1
        num2 //= 10

    # If not exactly 0 digits left, num1 and num2 cannot be permutations
    for digit_count in digits:
        if (digit_count):
            return False

    return True


def ProjectEuler_CyclicalFigurateNumbers_61() -> int:
    figurate_nums = defaultdict(list)
    result = -1

    difference = 0
    incrementer = 0
    n = 150

    # Calculate and store all four digit figurate numbers
    for i in range(1, n):
        curr_num = int((i*(i+1))/2)

        for j in range(3, 9):
            # Prune four-digit numbers with a '0' as their third digit
            if (curr_num >= 1000) and (curr_num <= 9999) and ((curr_num % 100) >= 10):
                figurate_nums[curr_num//100].append((curr_num % 100, j))
            curr_num += difference

        incrementer += 1
        difference += incrementer

    # Define function that will recursively search for cyclical numbers in 'figurate_nums'
    # Returns set of 6 cyclical numbers, or None if no set of 6 numbers if found
    def find_cyclical_nums(init_start, new_start: int, new_end: int, new_type: str, num_set: set, type_set: set) -> set:
        new_num = (new_start * 100) + new_end

        if (new_num in num_set) or (new_type in type_set):
            return None

        new_num_set = num_set.copy()
        new_num_set.add(new_num)

        new_type_set = type_set.copy()
        new_type_set.add(new_type)

        if (len(new_num_set) == 6):
            if (new_end == init_start):
                return new_num_set
            else:
                return None

        next_start = new_end
        next_ends = figurate_nums.get(next_start, None)
        if (next_ends == None):
            return None

        for next_end, next_type in next_ends:
            res = find_cyclical_nums(init_start, next_start, next_end, next_type, new_num_set, new_type_set)
            if (res != None):
                return res

        return None

    # Call recursive function with all possible starting values for set of cyclical nums
    for init_start, init_ends in figurate_nums.items():
        for init_end, init_type in init_ends:
            cyclical_nums = find_cyclical_nums(init_start, init_start, init_end, init_type, set(), set())
            if (cyclical_nums != None):
                result = sum(cyclical_nums)

    return result


def ProjectEuler_CubicPermutations_62() -> int:
    n = 10000
    k = 5

    permutation_count = dict()
    result = -1

    for i in range(1, n):
        curr_cube = i ** 3

        cube_digits = list(str(curr_cube))
        cube_digits.sort()
        cube_digits = tuple(cube_digits)

        if (cube_digits not in permutation_count):
            permutation_count[cube_digits] = [curr_cube, 1]
        else:
            permutation_count[cube_digits][1] += 1

        if (permutation_count[cube_digits][1] == k):
            result = permutation_count[cube_digits][0]
            break

    return result


def ProjectEuler_PowerfulDigitCounts_63() -> int:
    n = 25
    result = 0

    for exp in range(n+1):
        for base in range(1, 10):
            if (len(str(base ** exp)) == exp):
                result += 1

    return result


def ProjectEuler_OddPeriodSquareRoots_64() -> int:
    n = 10000
    num_odd_period_sqrt = 0

    for i in range(2, n+1):
        # Constants for getting continued fractions of i
        sqr_root = math.sqrt(i)
        a0 = math.floor(sqr_root)

        # If i is a perfect square, skip it as there is no irrational sqrt to approximate
        if (sqr_root == a0):
            continue

        # Initial numerator and denominator values
        curr_num = 1
        curr_denom = a0      # technically -a0, but set to +a0 for simpler code

        # Initialize set for storing num/denom constants at each iteration
        consts_set = set()

        # While have not found sequence of repetition for continued fractions
        while ((curr_num, curr_denom) not in consts_set):
            consts_set.add((curr_num, curr_denom))
            curr_num, curr_denom, _ = get_next_continued_fraction(i, sqr_root, curr_num, curr_denom)

        # Get period length of repetition sequence. If odd, increment counter
        period_len = len(consts_set)
        if ((period_len % 2) == 1):
            num_odd_period_sqrt += 1

    return num_odd_period_sqrt


def ProjectEuler_ConvergentsEulersNumber_65() -> int:
    n = 100
    e_sequence = []
    sequence_const = 2

    # Get partial fraction sequence for 'e'
    for i in range(1,n):
        if ((i % 3) == 2):
            k = math.ceil(i/3)
            e_sequence.append(2*k)
        else:
            e_sequence.append(1)

    # Initialize numerator and denominator to last (most nested) fraction value
    numerator = 1
    denominator = e_sequence[-1]

    # Solve all fractions from last to first
    for i in range(len(e_sequence)-2, -1, -1):
        numerator += e_sequence[i] * denominator
        numerator, denominator = denominator, numerator

    numerator += sequence_const * denominator

    # Have calculated numerator for nth convergent, now simply find sum of its digits:
    digit_sum = 0
    while (numerator > 0):
        digit_sum += numerator % 10
        numerator //= 10

    return digit_sum


def ProjectEuler_DiophantineEquation_66() -> int:
    d_max = 1000
    max_x_val = -1
    result = -1

    for d in range(1, d_max+1):
        # Constants for getting continued fractions of d
        sqr_root = math.sqrt(d)
        a0 = math.floor(sqr_root)

        # Last 'a' constant in period will be equal to  (2 * a0)
        final_a_val = 2*a0

        # If d is a perfect square, skip it as there is no irrational sqrt to approximate
        if (sqr_root == a0):
            continue

        # Initial numerator and denominator values
        curr_num = 1
        curr_denom = a0      # technically -a0, but set to +a0 for simpler code

        # Initialize deques for storing num/denom constants at each iteration
        a_consts = list()

        # Solve and store constants for first two continued fraction iteraions
        curr_num, curr_denom, next_a = get_next_continued_fraction(d, sqr_root, curr_num, curr_denom)
        a_consts.append(next_a)

        # Get all constants in first period
        while (a_consts[-1] != final_a_val):
            curr_num, curr_denom, next_a = get_next_continued_fraction(d, sqr_root, curr_num, curr_denom)
            a_consts.append(next_a)

        # Store appropriate number of constants depending on if 'n' is even or odd
        n = len(a_consts)

        if ((n % 2) == 0):
            a_consts.pop()
        else:
            for _ in range(n-1):
                curr_num, curr_denom, next_a = get_next_continued_fraction(d, sqr_root, curr_num, curr_denom)
                a_consts.append(next_a)

        # Take all constants in list and calculate convergent fraction
        num_converg, _ = get_convergent_fraction(a0, a_consts)

        # Numerator represents solution for 'x', check if largest 'x' value found so far
        if (num_converg > max_x_val):
            max_x_val = num_converg
            result = d

    return result


def ProjectEuler_MaximumPathSumII_67() -> int:
    try:
        input_file = open(Path("input_files/0067_triangle.txt"), "r")
        triangle = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find input 100 row triangle")
        return -1

    # Num rows in triangle equals num columns in triangle
    n = len(triangle)

    # Initialize dynamic programming array
    dp = [0] * n

    # Set base case in DP array
    for i in range(n):
        dp[i] = triangle[n-1][i]

    # Solve all subproblems in triangle from bottom row to top row
    for i in range(n-2, -1, -1):
        for j in range(len(triangle[i])):
            dp[j] = max(dp[j], dp[j+1]) + triangle[i][j]

    return dp[0]


def ProjectEuler_Magic5GonRing_68() -> int:
    combos = permutations([1,2,3,4,5,6,7,8,9,10],10)
    maximum_digit_string = -1

    for combo in combos:
        # Ensure clockwise string representation starts from index 0
        if (combo[0] > combo[2]) or (combo[0] > combo[4]) or (combo[0] > combo[6]) or (combo[0] > combo[8]):
            continue

        # Ensure '10' is in the outer ring so we get a 16-digit string instead of a 17-digit string
        if (combo[2] != 10) and (combo[4] != 10) and (combo[6] != 10) and (combo[8] != 10):
            continue

        # Get sum for line1
        sum1 = combo[0]+combo[1]+combo[3]

        # Check line1 against line2
        sum2 = combo[2]+combo[3]+combo[5]
        if (sum1 != sum2):
            continue

        # Check line1 against line3
        sum3 = combo[4]+combo[5]+combo[7]
        if (sum1 != sum3):
            continue

        # Check line1 against line4
        sum4 = combo[6]+combo[7]+combo[9]
        if (sum1 != sum4):
            continue

        # Check line1 against line5
        sum5 = combo[8]+combo[9]+combo[1]
        if (sum1 != sum5):
            continue

        # Found a valid magic 5-gon ring. Now, get 16-digit string representation:
        digit_string = combo[0]
        digit_append = [combo[1], combo[3], combo[2], combo[3], combo[5], combo[4], combo[5],
                        combo[7], combo[6], combo[7], combo[9], combo[8], combo[9], combo[1]]

        for dig in digit_append:
            digit_string = concatenate_integers(digit_string, dig)

        # Check if this is the largest string found so far
        maximum_digit_string = max(maximum_digit_string, digit_string)

    return maximum_digit_string


def ProjectEuler_TotientMaximum_69() -> int:
    # Want max ratio of i/phi(i) for all i <= n
    n = 1000000
    prime_val = 2
    solution = 1

    # Essentially looking for max value (within 'n') that has most distinct prime factors
    # So, simply multiply all smallest prime numbers together until reach threshold set by 'n'
    while ((solution * prime_val) <= n):
        if (check_prime(prime_val)):
            solution *= prime_val
        prime_val += 1

    return solution


def ProjectEuler_TotientPermutation_70() -> int:
    # Only need to check primes less than 10,000
    # Can have primes > 10,000 combined with other prime factors that still lead to val < n
    # However, this would require combining with smaller prime factors such as 2, 3, 5, etc.
    # And these smaller prime factors will greatly reduce phi which is undesirable
    try:
        input_file = open(Path("precomputed_primes/primes_10_thousand.txt"), "r")
        primes = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find list of prime numbers")
        return -1

    n_max = 10000000
    min_ratio = math.inf
    min_ratio_val = -1

    # Assume p1, p2 will not be small primes as this will greatly reduce phi
    # So, start searching from 100th prime onwards
    min_prime_idx = 100

    # Assume solution will have two prime factors only
    # i.e. (soln = p1^1 * p2^1) or (soln = p1^2)
    for i in range(min_prime_idx, len(primes)):
        for j in range(i, len(primes)):
            curr_n = primes[i] * primes[j]
            if (curr_n > n_max):
                break

            curr_phi, curr_ratio = get_phi_and_phi_ratio(curr_n, primes[i], primes[j])

            if (curr_ratio < min_ratio) and (isPermutation(curr_n, curr_phi)):
                min_ratio = curr_ratio
                min_ratio_val = curr_n

    return min_ratio_val


def main():
    sol_61 = ProjectEuler_CyclicalFigurateNumbers_61()
    print(f"sol_61 = {sol_61}")

    sol_62 = ProjectEuler_CubicPermutations_62()
    print(f"sol_62 = {sol_62}")

    sol_63 = ProjectEuler_PowerfulDigitCounts_63()
    print(f"sol_63 = {sol_63}")

    sol_64 = ProjectEuler_OddPeriodSquareRoots_64()
    print(f"sol_64 = {sol_64}")

    sol_65 = ProjectEuler_ConvergentsEulersNumber_65()
    print(f"sol_65 = {sol_65}")

    sol_66 = ProjectEuler_DiophantineEquation_66()
    print(f"sol_66 = {sol_66}")

    sol_67 = ProjectEuler_MaximumPathSumII_67()
    print(f"sol_67 = {sol_67}")

    sol_68 = ProjectEuler_Magic5GonRing_68()
    print(f"sol_68 = {sol_68}")

    sol_69 = ProjectEuler_TotientMaximum_69()
    print(f"sol_69 = {sol_69}")

    sol_70 = ProjectEuler_TotientPermutation_70()
    print(f"sol_70 = {sol_70}")


if __name__ == "__main__":
    main()
