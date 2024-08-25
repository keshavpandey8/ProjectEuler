# Keshav Pandey
from itertools import permutations
from collections import deque
import math

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


# Helper function that determines if input string is palindromic
# This palindrome checker uses two pointer algorithm
def check_palindrome(input_str: str) -> bool:
    for i in range(len(input_str) // 2):
        if (input_str[i] != input_str[-1-i]):
            return False
    return True


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
    combos = permutations(["1","2","3","4","5","6","7","8","9"], 9)
    pandigital_products = set()

    for combo in combos:
        for i in range(1, 8):
            for j in range(i+1, 9):
                multiplicand = int("".join(combo[:i]))
                multiplier = int("".join(combo[i:j]))
                product = int("".join(combo[j:]))

                if ((multiplicand * multiplier) == product):
                    pandigital_products.add(product)
                elif ((multiplicand * multiplier) > product):
                    break

    result = 0
    for pandigital_product in pandigital_products:
        result += pandigital_product

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

        while (curr_val > 0):
            curr_digit = curr_val % 10
            curr_val = curr_val // 10
            curr_sum += math.factorial(curr_digit)

        if (curr_sum == i):
            factorial_sum += i

    return factorial_sum


def ProjectEuler_CircularPrimes_35() -> int:
    num_circular_primes = 13            # We are told there are 13 circular primes below 100
    even_digits = {0, 2, 4, 6, 8}       # Set containing all even digits
    n = 1000000                         # Looking for all circular primes less than 'n'

    for i in range(101, n, 2):
        # If 'i' contains any even digits, cannot be circular prime
        curr_val = i

        while (curr_val > 0):
            curr_digit = curr_val % 10
            curr_val = curr_val // 10

            if (curr_digit in even_digits) or (curr_digit == 5):
                break
        else:
            curr_val = i
            rotate_val = deque(str(curr_val))
            num_digits = len(rotate_val)
            circular = True

            for _ in range(num_digits):
                if (not check_prime(curr_val)):
                    circular = False
                    break
                else:
                    rotate_val.appendleft(rotate_val.pop())
                    curr_val = int("".join(rotate_val))

            if (circular):
                num_circular_primes += 1

    return num_circular_primes


def ProjectEuler_DoubleBasePalindromes_36() -> int:
    n = 1000000
    sum_palindromes = 0

    # Assumption: only considering positive numbers
    for i in range(1, n):
        if check_palindrome(str(i)) and check_palindrome(str(bin(i))[2:]):
            sum_palindromes += i

    return sum_palindromes


def ProjectEuler_TruncatablePrimes_37() -> int:
    n = 1000000
    even_digits = {0, 2, 4, 6, 8}
    sum_primes = 0

    for i in range(11, n, 2):
        # If 'i' contains even digits (aside from leftmost digit) -> cannot be truncatable prime
        curr_val = i
        while (curr_val > 9):
            curr_digit = curr_val % 10
            curr_val = curr_val // 10

            if (curr_digit in even_digits) or (curr_digit == 5):
                break
        else:
            # Check if 'i' is prime when truncating rightmost digits
            curr_val = i
            while (curr_val > 0):
                if (check_prime(curr_val)):
                    curr_val = curr_val // 10
                else:
                    break
            else:
                # Check if 'i' is prime when truncating leftmost digits
                truncate = deque(str(i))
                truncate.popleft()

                while (len(truncate) > 0):
                    curr_val = int("".join(truncate))
                    if (check_prime(curr_val)):
                        truncate.popleft()
                    else:
                        break
                else:
                    sum_primes += i

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
    p_max = 1000

    max_solutions = 0
    best_p_val = -1

    for p in range(3, p_max+1):
        curr_solns = 0
        for a in range(1, p):
            a_squared = math.pow(a, 2)
            for b in range(a, p):
                c = math.sqrt(a_squared + math.pow(b, 2))
                if ((a + b + c) == p):
                    curr_solns += 1
                elif ((a + b + c) > p):
                    break
        if (curr_solns > max_solutions):
            max_solutions = curr_solns
            best_p_val = p

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


if __name__ == "__main__":
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
