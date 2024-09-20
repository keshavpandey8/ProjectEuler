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


# TODO: solve using mathematical properties
def ProjectEuler_Multiples_1() -> int:
    # Consider all multiples up to 1000
    n = 1000
    curr_sum = 0

    # Iterate through all values up to limit
    for i in range(n):
        # Update sum if current value is a multiple of 3 or 5
        if ((i % 3) == 0) or ((i % 5) == 0):
            curr_sum += i

    return curr_sum


# TODO: can rewrite to make use of Golden Ratio
def ProjectEuler_EvenFibonacci_2() -> int:
    # Consider values in sequence less than 4,000,000
    n = 4000000

    # Starting values in Fibonacci Sequence
    term1 = 1
    term2 = 2
    new_term = 3

    # Current sum of all even number
    curr_sum = 2

    # While newest term in sequence is less than limit
    while (new_term < n):
        # Update previous terms
        term1, term2 = term2, new_term

        # Update current sum
        if ((new_term % 2) == 0):
            curr_sum += new_term

        # Calculate new value
        new_term = term1 + term2

    return curr_sum


def ProjectEuler_LargestPrimeFactor_3() -> int:
    # Input value for which we are finding largest prime factor
    num = 600851475143

    # Start checking divisors at i=3 since 'num' is not even
    i = 3
    lastPrime = 1

    # Note: there can be (at most) one prime factor of num that is > sqrt(num):
    limit = math.floor(math.sqrt(num))

    # Iterate over all divisors until limit is reached, or all divisors are found
    while (num > 1) and (i <= limit):
        while ((num % i) == 0):
            num //= i
        else:
            limit = math.floor(math.sqrt(num))
            lastPrime = i
        i += 2

    # If 'num' is still greater than 1
    # Then it must be equal to the last (largest) prime factor of original input val
    if (num > 1):
        lastPrime = num

    return lastPrime


def ProjectEuler_LargestPalindromeProduct_4() -> int:
    # Helper function to see if input val is a product of two 3-digit integers
    # Only multiples of 11 are checked, as 'val' must be divisible by 11:
    def is_product_of_3_digit_nums(val: int) -> bool:
        for i in range(110, 1000, 11):
            # As i increases, curr_res will decrease.
            # So, if curr_res already smaller than 3 digits, we can quit
            curr_res = val / i
            if (curr_res < 100):
                return False

            # Check if current i value meets criteria
            if ((val % i) == 0) and (curr_res <= 999):
                return True

        # Return False if all 3 digit values are checked and no valid solution is found
        return False

    # Assume solution is 6 digits long
    # Therefore it must be of form: 'abccba' where a,b,c are one-digit numbers
    # Check all possible values from largest to smallest:
    for a in range(9, -1, -1):
        for b in range(9, -1, -1):
            for c in range(9, -1, -1):
                val = 11 * (9091*a + 910*b + 100*c)
                if (is_product_of_3_digit_nums(val)):
                    return val

    # No solution found, return -1
    return -1


def ProjectEuler_SmallestMultiple_5() -> int:
    # Find smallest multiple of all integers from 0 to 20
    num = 20
    prime_factors = list()
    solution = 1

    # Get all prime numbers from 2 to 20
    for i in range(2, num+1):
        if (check_prime(i)):
            solution *= i
            prime_factors.append(i)

    # Iterate through all values up to num
    for i in range(2, num+1):
        # If current solution value is already a multiple of i, then skip to next i value
        if ((solution % i) == 0):
            continue

        # Find a prime factor that "solution" must be multipled with to make it a multiple of i
        for prime_val in prime_factors:
            if ((solution * prime_val) % i) == 0:
                solution *= prime_val
                break

    return solution


def ProjectEuler_SumSquareDifference_6() -> int:
    # Find solution for first 100 natural numbers
    num = 100

    # Sum from 1...n = (n)(n+1)/2
    square_of_sums = math.pow(((num * (num + 1)) / 2), 2)

    # Sum of 1^2...n^2 = (n)(n+1)(2n+1)/6
    sum_of_squares = ((num)*(num + 1)*((2*num)+1)) / 6

    # Subtract values to get final result
    solution = int(square_of_sums - sum_of_squares)
    return solution


def ProjectEuler_10001stPrime_7() -> int:
    # Find 10001st prime number. Skipping '2' so set num=10000
    num = 10000
    curr_val = 1

    while (num > 0):
        curr_val += 2
        if (check_prime(curr_val)):
            num -= 1

    return curr_val


def ProjectEuler_LargestProductInSeries_8() -> int:
    # Read 1000 digit input number
    try:
        input = open(Path("input_files/0008_num.txt"), "r")
        num = input.readline()
        input.close()
    except FileNotFoundError:
        print(f"Error: could not find 1000 digit input number")
        return -1

    # Find largest product from 'k' adjacent digits in input number
    largest_product = -1
    k = 13
    n = len(num)

    # Iterate through ith digit one-by-one
    i = 0
    calcNewProduct = True
    curr_product = 1

    while (i < n):
        # If at start of list or just encountered a 0, must calculate product of next 'k' vals
        if (calcNewProduct):
            curr_product = 1
            num_vals = k

            while (num_vals > 0) and (i < n):
                curr_digit = int(num[i])
                if (curr_digit == 0):
                    curr_product = 1
                    num_vals = k
                    i += 1
                    continue
                else:
                    curr_product *= curr_digit
                    num_vals -= 1
                    i += 1

            # Only update if found product of 13 adjacent digits
            # (edge case: quit prev. loop because (i >= n), not because new product found)
            if (num_vals == 0):
                largest_product = max(curr_product, largest_product)
            calcNewProduct = False

        # If already have product of prev. 'k' vals
        else:
            # If this digit is a 0, we reset and calculate product of next 'k' digits
            curr_digit = int(num[i])
            if (curr_digit == 0):
                calcNewProduct = True
                i += 1
            else:
                # Multiply next val and divide off prev. val to get next 'k' digit product
                curr_product = (curr_product / int(num[i - k])) * curr_digit
                largest_product = max(curr_product, largest_product)
                i += 1

    return int(largest_product)


# TODO: solve mathetmatically
def ProjectEuler_SpecialPythagoreanTriplet_9() -> int:
    # Searching for triplet that sums to target value of 1000
    target = 1000
    solution = -1

    # Iterate through all possible a,b combinations such that a < b
    for a in range(1, 1001):
        for b in range(a, 1001):
            # Calculate 'c' such that sum of a,b,c equals target
            c = target - a - b

            # If this c value is less than b, all future c values will also be less than b
            if (c < b):
                break

            # If all criteria is met, calculate solution and return
            if ((a + b + c) == target) and (((a*a) + (b*b)) == (c*c)):
                solution = a * b * c
                return solution

    return solution


def ProjectEuler_SummationPrimes_10() -> int:
    # Find sum of all primes less than n
    n = 2000000
    prime_sum = 2

    # Initialize sieve
    prime_sieve = [True] * n
    prime_sieve[0] = False
    prime_sieve[1] = False

    # Skip all even numbers after 2, as they cannot be prime
    for i in range(3, n, 2):
        # If current num has not been eliminated, it must be prime
        if (prime_sieve[i]):
            prime_sum += i

            # Eliminate all odd multiples of current num in sieve
            for val in range(i*3, n, i*2):
                prime_sieve[val] = False

    return prime_sum


def main():
    sol_1 = ProjectEuler_Multiples_1()
    print(f"sol_1 = {sol_1}")

    sol_2 = ProjectEuler_EvenFibonacci_2()
    print(f"sol_2 = {sol_2}")

    sol_3 = ProjectEuler_LargestPrimeFactor_3()
    print(f"sol_3 = {sol_3}")

    sol_4 = ProjectEuler_LargestPalindromeProduct_4()
    print(f"sol_4 = {sol_4}")

    sol_5 = ProjectEuler_SmallestMultiple_5()
    print(f"sol_5 = {sol_5}")

    sol_6 = ProjectEuler_SumSquareDifference_6()
    print(f"sol_6 = {sol_6}")

    sol_7 = ProjectEuler_10001stPrime_7()
    print(f"sol_7 = {sol_7}")

    sol_8 = ProjectEuler_LargestProductInSeries_8()
    print(f"sol_8 = {sol_8}")

    sol_9 = ProjectEuler_SpecialPythagoreanTriplet_9()
    print(f"sol_9 = {sol_9}")

    sol_10 = ProjectEuler_SummationPrimes_10()
    print(f"sol_10 = {sol_10}")


if __name__ == "__main__":
    main()
