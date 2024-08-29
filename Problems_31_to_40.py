# Keshav Pandey
from collections import deque
import json
import math
from pathlib import Path

# Returns true if 'val' is a 1 to 9 pandigital
def pandigital(val: int) -> bool:
    result = 0

    while (val > 0):
        result |= 1 << ((val % 10))
        val //= 10

    # If pandigital must have (result == 0b1111111110)
    return (result == 0x3FE)


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


# Helper function that determines if integer 'val' is palindromic in base 2
def check_palindrome(val: int) -> bool:
    reverse = 0
    temp = val

    while (temp > 0):
        reverse = (reverse << 1) | (temp & 1)
        temp >>= 1

    return (val == reverse)


def ProjectEuler_CoinSums_31() -> int:
    # Value of UK coins in pence
    coins = [1, 2, 5, 10, 20, 50, 100, 200]
    num_coins = len(coins)

    # Initialize variables:
    target = 200
    dp = [[0] * num_coins for _ in range(target + 1)]

    # Solve all sub-problems in bottom-up approach:
    for curr_total in range(1, target + 1):
        curr_combos = 0
        for i in range(num_coins):
            coin = coins[i]

            if (curr_total - coin) > 0:
                curr_combos += dp[curr_total-coin][i]
            elif (curr_total - coin) == 0:
                curr_combos += 1

            dp[curr_total][i] = curr_combos

    total_combos = dp[target][num_coins - 1]
    return total_combos


def ProjectEuler_PandigitalProducts_32() -> int:
    result = 0
    pandigital_products = set()

    # Try all (1-digit * 4-digit = 4-digit) combinations
    for i in range(1, 10):
        for j in range(1000, 10000):
            product = i*j
            if (product > 9999):
                break

            pan_val = j*100000 + product*10 + i

            if (pandigital(pan_val)) and (product not in pandigital_products):
                pandigital_products.add(product)
                result += product

    # Try all (2-digit * 3-digit = 4-digit) combinations
    for i in range(10, 100):
        for j in range(100, 1000):
            product = i*j
            if (product > 9999):
                break

            pan_val = j*1000000 + product*100 + i

            if (pandigital(pan_val)) and (product not in pandigital_products):
                pandigital_products.add(product)
                result += product

    return result


def ProjectEuler_DigitCancellingFractions_33() -> int:
    num_prod = 1
    denom_prod = 1

    for num in range(11, 100):
        # Skip current num value if it contains repeated digits or is trivial
        if ((num % 11) == 0) or ((num % 10) == 0):
            continue

        for denom in range(num+1, 100):
            # Skip current denom value if it contains repeated digits or is trivial
            if ((denom % 11) == 0) or ((denom % 10) == 0):
                continue

            # If num/denom pair does not have one repeated digit to "cancel", skip it
            if len(set(str(num) + str(denom))) != 3:
                continue

            num_set = set([num//10, num%10])
            denom_set = set([denom//10, denom%10])
            intersection = num_set & denom_set
            repeated_digit = int(intersection.pop())

            if (num // 10) == repeated_digit:
                new_num = num % 10
            else:
                new_num = num // 10

            if (denom // 10) == repeated_digit:
                new_denom = denom % 10
            else:
                new_denom = denom // 10

            actual_fraction = num / denom
            new_fraction = new_num / new_denom

            if (actual_fraction == new_fraction):
                num_prod *= new_num
                denom_prod *= new_denom

    result = denom_prod // math.gcd(num_prod, denom_prod)
    return result


def ProjectEuler_DigitFactorials_34() -> int:
    n = 50000
    factorial_sum = 0

    for i in range(10, n+1):
        curr_val = i
        curr_sum = 0

        while (curr_val > 0) and (curr_sum <= i):
            curr_sum += math.factorial(curr_val % 10)
            curr_val //= 10

        if (curr_sum == i):
            factorial_sum += i

    return factorial_sum


def ProjectEuler_CircularPrimes_35() -> int:
    try:
        input_file = open(Path("precomputed_primes/primes_1_million.txt"), "r")
        primes_list = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find list of prime numbers")
        return -1

    num_circular_primes = 13            # We are told there are 13 circular primes below 100
    primes_list = primes_list[25:]      # Ignore primes below 100
    bad_digits = {0, 2, 4, 6, 8, 5}     # All digits which disqualify num from being circular prime

    for p in primes_list:
        # If 'p' contains any even digits or 5, cannot be circular prime
        curr_val = p // 10

        while (curr_val > 0):
            curr_digit = curr_val % 10
            curr_val //= 10

            if (curr_digit in bad_digits):
                break
        else:
            curr_val = p
            rotate_val = deque(str(curr_val))
            num_digits = len(rotate_val)

            for _ in range(num_digits):
                if (not check_prime(curr_val)):
                    break
                else:
                    rotate_val.appendleft(rotate_val.pop())
                    curr_val = int("".join(rotate_val))
            else:
                num_circular_primes += 1

    return num_circular_primes


def ProjectEuler_DoubleBasePalindromes_36() -> int:
    sum_palindromes = 0

    # Check if base 10 palindromes of form 'a' and 'aa' are palindromic in base 2
    for a in range(1, 10, 2):
        if check_palindrome(a):
            sum_palindromes += a

        palindrome = (a*10)+a
        if check_palindrome(palindrome):
            sum_palindromes += palindrome

    # Check if base 10 palindromes of form 'aba' and 'abba' are palindromic in base 2
    for a in range(1, 10, 2):
        for b in range(10):
            palindrome = (a*100) + (b*10) + a
            if check_palindrome(palindrome):
                sum_palindromes += palindrome

            palindrome = (a*1000) + (b*100) + (b*10) + a
            if check_palindrome(palindrome):
                sum_palindromes += palindrome

    # Check if base 10 palindromes of form 'abcba' and 'abccba' are palindromic in base 2
    for a in range(1, 10, 2):
        for b in range(10):
            for c in range(10):
                palindrome = (a*10000) + (b*1000) + (c*100) + (b*10) + a
                if check_palindrome(palindrome):
                    sum_palindromes += palindrome

                palindrome = (a*100000) + (b*10000) + (c*1000) + (c*100) + (b*10) + a
                if check_palindrome(palindrome):
                    sum_palindromes += palindrome

    return sum_palindromes


def ProjectEuler_TruncatablePrimes_37() -> int:
    try:
        input_file = open(Path("precomputed_primes/primes_1_million.txt"), "r")
        primes_list = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find list of prime numbers")
        return -1

    sum_primes = 0                      # Store sum of all trunc. primes
    total_trunc_primes = 11             # We are told there are only 11 trunc. primes
    primes_list = primes_list[4:]       # Ignore single digit primes
    bad_digits = {0, 2, 4, 6, 8, 5}     # All digits which disqualify num from being trunc. prime

    for p in primes_list:
        # Found all 11 truncatable primes, exit loop
        if (total_trunc_primes == 0):
            break

        # If 'p' contains even digits or 5 (aside from leftmost digit), cannot be trunc. prime
        curr_val = p // 10

        while (curr_val > 9):
            curr_digit = curr_val % 10
            curr_val //= 10

            if (curr_digit in bad_digits):
                break
        else:
            # Check if 'p' is prime when truncating rightmost digits
            curr_val = p
            while (curr_val > 0):
                if (check_prime(curr_val)):
                    curr_val = curr_val // 10
                else:
                    break
            else:
                # Check if 'p' is prime when truncating leftmost digits
                truncate = deque(str(p))
                truncate.popleft()

                while (len(truncate) > 0):
                    curr_val = int("".join(truncate))
                    if (check_prime(curr_val)):
                        truncate.popleft()
                    else:
                        break
                else:
                    sum_primes += p
                    total_trunc_primes -= 1

    return sum_primes


def ProjectEuler_PandigitalMultiples_38() -> int:
    n = 10000
    largest_pandigital_num = 0
    pandigital_digits = {"1","2","3","4","5","6","7","8","9"}

    for i in range(1, n):
        curr_pandigital = str(i)
        multiplier = 2

        while (len(curr_pandigital) < 9):
            curr_val = i * multiplier
            curr_pandigital += str(curr_val)
            multiplier += 1

        if (len(curr_pandigital) == 9):
            if (set(curr_pandigital) == pandigital_digits):
                if (int(curr_pandigital) > largest_pandigital_num):
                    largest_pandigital_num = int(curr_pandigital)

    return largest_pandigital_num


def ProjectEuler_IntegerRightTriangles_39() -> int:
    # Store the number of distinct solutions for all p <= p_max
    p_max = 1000
    p_solns = [0] * (p_max+1)

    # Initialize variables to store final solution
    max_solutions = 0
    best_p_val = -1

    # Find all integer right angle triangles with perimeters <= p_max
    for a in range(1, p_max+1):
        a_squared = a ** 2
        for b in range(a, p_max+1):
            c = (math.sqrt(a_squared + (b ** 2)))
            if ((a + b + c) > p_max):
                break

            if (int(c) == c):
                perimeter = a+b+int(c)
                p_solns[perimeter] += 1

    # Find perimeter with most solutions, then return result
    for i in range(p_max+1):
        if (p_solns[i] > max_solutions):
            max_solutions = p_solns[i]
            best_p_val = i

    return best_p_val


def ProjectEuler_ChampernowneConstant_40() -> int:
    n = 1000000
    desired_indices = [1, 10, 100, 1000, 10000, 100000, 1000000]
    result = 1

    irrational_fraction = ""
    current_val = 1

    while (len(irrational_fraction) < n):
        irrational_fraction += str(current_val)
        current_val += 1

    for idx in desired_indices:
        result *= int(irrational_fraction[idx-1])

    return result


def main():
    sol_31 = ProjectEuler_CoinSums_31()
    print(f"sol_31 = {sol_31}")

    sol_32 = ProjectEuler_PandigitalProducts_32()
    print(f"sol_32 = {sol_32}")

    sol_33 = ProjectEuler_DigitCancellingFractions_33()
    print(f"sol_33 = {sol_33}")

    sol_34 = ProjectEuler_DigitFactorials_34()
    print(f"sol_34 = {sol_34}")

    sol_35 = ProjectEuler_CircularPrimes_35()
    print(f"sol_35 = {sol_35}")

    sol_36 = ProjectEuler_DoubleBasePalindromes_36()
    print(f"sol_36 = {sol_36}")

    sol_37 = ProjectEuler_TruncatablePrimes_37()
    print(f"sol_37 = {sol_37}")

    sol_38 = ProjectEuler_PandigitalMultiples_38()
    print(f"sol_38 = {sol_38}")

    sol_39 = ProjectEuler_IntegerRightTriangles_39()
    print(f"sol_39 = {sol_39}")

    sol_40 = ProjectEuler_ChampernowneConstant_40()
    print(f"sol_40 = {sol_40}")


if __name__ == "__main__":
    main()
