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
    solution = -1

    # No possible solution if num is negative
    if (num < 0):
        return solution

    # Largest prime factor of num must be less than or equal to sqrt(num)
    # TODO: the above comment is wrong and this is actually incorrect
    # TODO: read overview PDF for this problem for details...
    max_prime_factor = math.floor(math.sqrt(num))

    # Iterate through all possible prime factors from largest to smallest
    # The first value that evenly divides num and is prime will be our solution
    for i in range(max_prime_factor, 1, -1):
        if ((num % i) == 0) and (check_prime(i)):
            solution = i
            break

    return solution


# TODO: solve using mathematical properties
def ProjectEuler_LargestPalindromeProduct_4() -> int:
    # Min/Max bounds for largest palindrome that is product of two 3-digit values
    min_val = 100*100
    max_val = 999*999
    solution = -1

    # Helper function that determines if input val is palindromic
    def isPalindrome(val: int) -> bool:
        val = str(val)
        n = len(val)

        for i in range(0, math.floor(n / 2)):
            if (val[i] != val[n-1-i]):
                return False
        return True
    
    # Helper function that determines if input val is a product of two 3-digit integers
    def is_product_of_3_digit_nums(val: int) -> bool:
        for i in range(100, 1000):
            # As i increases, curr_res will decrease. 
            # So, if curr_res already smaller than 3 digits, we can quit
            curr_res = val / i
            if (curr_res < 100):
                return False
            
            # Check if current i value meets criteria
            if ((val % i) == 0) and (curr_res >= 100) and (curr_res <= 999):
                return True

        # Return False if all 3 digit values are checked and no valid solution is found
        return False

    # Check all values within min/max bounds from largest to smallest
    # First value found that meets all criteria is our solution
    for i in range(max_val, min_val-1, -1):
        if (isPalindrome(i)) and (is_product_of_3_digit_nums(i)):
            solution = i
            break

    return solution


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
    # Find 10001st prime number
    num = 10001
    curr_val = 1

    while (num > 0):
        curr_val += 1
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

# TODO: resolve using sieve method
def ProjectEuler_SummationPrimes_10() -> int:
    # Find sum of all primes less than num
    num = 2000000

    # Store all found primes in set
    # We already include "2" and "3", so we can skip all values that are multiples of 2 and 3
    prime_numbers = list([2, 3])
    sum = 5

    # Iterate through all odd numbers from 5 to num. Check if each number is prime or not
    # for i in range(3, num, 2):
    i = 5
    increment2 = True
    while (i < num):
        # Calculate max possible prime factor for current num 
        max_prime = math.floor(math.sqrt(i))
        isPrime = True

        # Loop through all possible prime factors for i
        # If no factors are found, then current i value is prime
        for prime_val in prime_numbers:
            if (prime_val > max_prime):
                break
            elif ((i % prime_val) == 0):
                isPrime = False
                break

        # Update sum and list of prime numbers if necessary
        if (isPrime):
            prime_numbers.append(i)
            sum += i

        # Want to alternate incrementing i by 2 and by 4
        # This is so we skip over all elements that are divisible by 2 and 3
        if (increment2):
            i += 2
            increment2 = False
        else:
            i += 4
            increment2 = True

    return sum


if __name__ == "__main__":
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
