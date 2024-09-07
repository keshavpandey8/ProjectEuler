# Keshav Pandey
import decimal
import json
import math
from pathlib import Path
import sys

# Input num must be a positive integer
def calc_digit_factorial(num: int, digit_factorials: list) -> int:
    result = 0
    while (num > 0):
        result += digit_factorials[num % 10]
        num //= 10

    return result


def ProjectEuler_OrderedFractions_71() -> int:
    # Searching for a fraction that is just less than 3/7
    upper_bound_num = 3
    upper_bound_denom = 7

    # Solution should be a fraction with denominator <= max_d
    max_d = 1000000

    # For fractions with denom <= 8, we are told the fraction to the left of 3/7 is 2/5
    result_num = 2
    result_denom = 5

    # Utilize Farey Series:
    farey_factor = (max_d - result_denom) // upper_bound_denom
    result_num += upper_bound_num * farey_factor
    result_denom += upper_bound_denom * farey_factor

    return result_num


def ProjectEuler_CountingFractions_72() -> int:
    # Count number of reduced proper fractions with denom <= max_d
    max_d = 1000000
    fraction_count = 0

    # Initialize totient array to value of its index (t[i] = i)
    totients = [i for i in range(max_d+1)]

    # Sum totients for all denom <= max_d
    for i in range(2, max_d+1):
        if (totients[i] != i):
            # i is a composite number, increment fraction_count by phi(i)
            fraction_count += totients[i]
        else:
            # i is a prime number, will have a reduced fraction for all (num < i)
            fraction_count += i-1

            # Account for all multiples of i
            phi_const = (i-1) / i
            for mult in range(i*2, max_d+1, i):
                totients[mult] *= phi_const

    return int(fraction_count)


# TODO: can optimize further
def ProjectEuler_CountingFractionsInRange_73() -> int:
    # Count number of reduced proper fractions with denom <= max_d
    max_d = 12000
    fraction_count = 0

    # Given lower bound fraction = 1/3
    a = 1
    b = 3

    # Given upper bound fraction = 1/2
    c0 = 1
    d0 = 2

    # Calculate fraction immediately to right of 1/3 in Farey series: (c/d)
    k = (max_d - d0) // b
    c = c0 + (k*a)
    d = d0 + (k*b)

    # While (c/d != upper bound), calculate next fraction in Farey Series
    while (c != c0) or (d != d0):
        fraction_count += 1

        # Calculate next fraction
        k = (max_d + b) // d
        e = k*c - a
        f = k*d - b

        # Update variables
        a, b = c, d
        c, d = e, f

    return fraction_count


def ProjectEuler_DigitFactorialChains_74() -> int:
    # Pre-calculate factorial value for each digit for faster iterations
    digit_factorials = [1] * 10
    for i in range(2, 10):
        digit_factorials[i] = i * digit_factorials[i-1]

    # Initialize variables to define problem
    n = 1000000
    k = 60
    num_k_chains = 0

    # Store all integers known to lead to infinite loop in memoization table
    memo = [-1] * 2180000

    # 1 (-> 1) ; 2 (-> 2)
    memo[1] = 1
    memo[2] = 1

    # 145 (-> 145) ; 40585 (-> 40585)
    memo[145] = 1
    memo[40585] = 1

    # 169 -> 363601 -> 1454 (-> 169)
    memo[169] = 3
    memo[363601] = 3
    memo[1454] = 3

    # 871 -> 45361 (-> 871)
    memo[871] = 2
    memo[45361] = 2

    # 872 -> 45362 (-> 872)
    memo[872] = 2
    memo[45362] = 2

    # Check every value up to 'n' for an exactly 'k' length chain
    for i in range(3, n):
        if (memo[i] != -1):
            continue

        curr_chain = list()
        curr_val = i

        while (memo[curr_val] == -1):
            curr_chain.append(curr_val)
            curr_val = calc_digit_factorial(curr_val, digit_factorials)

        init_chain_len = len(curr_chain)
        chain_len = init_chain_len + memo[curr_val]

        if (chain_len == k):
            num_k_chains += 1

        for j in range(init_chain_len):
            memo[curr_chain[j]] = chain_len - j

    return num_k_chains


def ProjectEuler_SingularIntegerRightTriangles_75() -> int:
    # Initialize constants
    max_L = 1500000
    m_limit = 875

    # The ith index represents wire length, the value at this index represents number of triangles
    # that can be formed with exactly this length
    lengths = [0] * (max_L+1)

    # Generate all primitive triangles with perimeters <= max_L
    for m in range(1, m_limit):
        if ((m % 2) == 0):
            start = 1
        else:
            start = 2

        for n in range(start, m, 2):
            if (math.gcd(m, n) != 1):
                continue

            perimeter = 2*m*(m+n)
            if (perimeter > max_L):
                break

            for p_mult in range(perimeter, max_L+1, perimeter):
                lengths[p_mult] += 1

    # Get number of wire lengths with exactly 1 triangle summing to that wire length
    result = 0
    for val in lengths:
        if (val == 1):
            result += 1

    return result


def ProjectEuler_CountingSummations_76() -> int:
    n = 101
    dp = [[0] * n for _ in range(n)]

    # Number of ways you can add up to 0 with numbers only as big as 'i' = 1
    for i in range(n):
        dp[0][i] = 1

    # Find total number of ways you can add up to 'i'
    for i in range(1, n):
        # Calculate number of ways you can add up to 'i' when only using values as big as 'max_sum_val'
        for max_sum_val in range(1, i+1):
            complement = i - max_sum_val
            dp[i][max_sum_val] = dp[complement][max_sum_val] + dp[i][max_sum_val-1]

        # Number of ways you can add up to 'i' using positive numbers no greater than 'i+1' is
        # equal to number of ways you can add up to 'i' using positive numbers no greater than 'i'
        for max_sum_val in range(i+1, n):
            dp[i][max_sum_val] = dp[i][max_sum_val-1]

    # Desired result is number of ways we can add up to 100 using integers only as large as 99
    return dp[100][99]


# Interesting note: for dp[i][-1] the summation is incorrect if 'i' is prime.
# The correct summation for these cases would be 1 less than what is stored in dp array
# This is because we count summation: i+0=i as a valid sum even though 0 is not prime
# We have coded solution like this for simplicity :)
def ProjectEuler_PrimeSummations_77() -> int:
    # TODO: actually only need primes less than 100....
    try:
        input_file = open(Path("precomputed_primes/primes_10_thousand.txt"), "r")
        primes_set = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find list of prime numbers")
        return -1

    n = 75
    k = 5000
    result = -1
    primes_set = set(primes_set)
    dp = [[0] * n for _ in range(n)]

    # Number of ways you can add up to 0 with numbers only as big as 'i' = 1
    for i in range(n):
        dp[0][i] = 1

    # Find total number of ways you can add up to 'i'
    for i in range(1, n):
        # Calculate number of ways you can add up to 'i' when only using primes as big as 'p'
        for p in range(1, i+1):
            if (p not in primes_set):
                dp[i][p] = dp[i][p-1]
                continue

            complement = i - p
            dp[i][p] = dp[complement][p] + dp[i][p-1]

        # Number of ways you can add up to 'i' using positive numbers no greater than 'i+1' is
        # equal to number of ways you can add up to 'i' using positive numbers no greater than 'i'
        for p in range(i+1, n):
            dp[i][p] = dp[i][p-1]

        if (dp[i][n-1] > k):
            result = i
            break

    return result


def ProjectEuler_CoinPartitions_78() -> int:
    max_n = 57500
    div_val = 1000000
    result = -1

    # Store and memoize reused calculation terms here:
    p2 = [0] * (max_n+1)
    p3 = [0] * (max_n+1)

    p2[1] = 1
    p3[1] = 2
    k1 = 4

    for k in range(2, max_n+1):
        p2[k] = p2[k-1] + k1
        p3[k] = p3[k-1] + k1 + 1
        k1 += 3

    # Note: number of partitions for ((n < 0) == 0) and for ((n = 0) == 1)
    memo = [0] * (max_n+1)
    memo[0] = 1

    # Iterate over values of 'n' until find an 'n' where (partitions(n) % div_val) == 0
    for n in range(1, max_n+1):
        curr_partitions = 0
        k = 1
        p1 = 1

        # Combine terms and accumulate current number of partitions
        while (n >= p3[k]):
            curr_partitions += p1 * (memo[n-p2[k]] + memo[n-p3[k]])
            k += 1
            p1 *= -1

        if (n >= p2[k]):
            curr_partitions += p1 * memo[n-p2[k]]

        # Save current number of partitions in memo table for future calculations
        memo[n] = curr_partitions % div_val
        if (memo[n] == 0):
            result = n
            break

    return result


# TODO: can also solve by building a graph and using topological sort (possibly better ??)
def ProjectEuler_PasscodeDerivation_79() -> int:
    try:
        input_file = open(Path("input_files/0079_keylog.txt"), "r")
        keylog = input_file.readlines()
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find input key log")
        return -1

    keys = list()
    for line in keylog:
        data = tuple(line.strip())
        keys.append(list(data))

    n = len(keys)
    solution = ""

    while (n > 0):
        last_vals = set()
        not_last_vals = set()

        for i in range(n):
            last_vals.add(keys[i][-1])
            for j in range(len(keys[i])-1):
                not_last_vals.add(keys[i][j])

        last_val = last_vals.difference(not_last_vals)
        if (len(last_val) != 1):
            print(f"Error: unexpected outcome...solution contains repeated digits")
            sys.exit()
        last_val = last_val.pop()

        idx = 0
        while (idx < n):
            if (keys[idx][-1] == last_val):
                keys[idx].pop()
                if (len(keys[idx]) == 0):
                    keys.pop(idx)
                    n -= 1
                    continue
            idx += 1

        solution += last_val

    solution = int(solution[::-1])
    return solution


# TODO: solve using a mathematical method (Newton Raphson?)
def ProjectEuler_SquareRootDigitalExpansion_80() -> int:
    n = 100
    total_sum = 0
    decimal.getcontext().prec = 102

    for i in range(1, n+1):
        if (int(math.sqrt(i)) == math.sqrt(i)):
            continue

        # We get extra decimal places ("prec = 102" above) then manually trucate them to avoid
        # last decimal value being rounded up and thus giving us an incorrect sum
        sqrt_i = str(decimal.Decimal(i).sqrt())
        sqrt_i = sqrt_i.replace(".", "")
        sqrt_i = sqrt_i[:100:]

        # Expect length of 100 decimal digits
        if (len(sqrt_i) != 100):
            print(f"Error: unexpected value found finding first 100 decimals of irrational num: {i}")
            print(f"{len(sqrt_i)}, {sqrt_i}")
            sys.exit()

        # Sum all digits in string
        for digit in sqrt_i:
            total_sum += int(digit)

    return total_sum


def main():
    sol_71 = ProjectEuler_OrderedFractions_71()
    print(f"sol_71 = {sol_71}")

    sol_72 = ProjectEuler_CountingFractions_72()
    print(f"sol_72 = {sol_72}")

    sol_73 = ProjectEuler_CountingFractionsInRange_73()
    print(f"sol_73 = {sol_73}")

    sol_74 = ProjectEuler_DigitFactorialChains_74()
    print(f"sol_74 = {sol_74}")

    sol_75 = ProjectEuler_SingularIntegerRightTriangles_75()
    print(f"sol_75 = {sol_75}")

    sol_76 = ProjectEuler_CountingSummations_76()
    print(f"sol_76 = {sol_76}")

    sol_77 = ProjectEuler_PrimeSummations_77()
    print(f"sol_77 = {sol_77}")

    sol_78 = ProjectEuler_CoinPartitions_78()
    print(f"sol_78 = {sol_78}")

    sol_79 = ProjectEuler_PasscodeDerivation_79()
    print(f"sol_79 = {sol_79}")

    sol_80 = ProjectEuler_SquareRootDigitalExpansion_80()
    print(f"sol_80 = {sol_80}")


if __name__ == "__main__":
    main()
