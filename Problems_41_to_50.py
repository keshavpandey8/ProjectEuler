# Keshav Pandey
from itertools import permutations
import json
import math
from pathlib import Path
from ProjectEuler_Helpers import check_prime

# Returns true if 'val' is a 0 to 9 pandigital
def pandigital(val: int) -> bool:
    result = 0

    while (val > 0):
        result |= 1 << ((val % 10))
        val //= 10

    # If pandigital must have (result == 0b1111111111)
    return (result == 0x3FF)


def ProjectEuler_PandigitalPrime_41() -> int:
    pandigital_prime = 0
    combos = permutations([1,2,3,4,5,6,7],7)

    for combo in combos:
        curr_val = ""
        for val in combo:
            curr_val += str(val)
        curr_val = int(curr_val)

        if (check_prime(curr_val)) and (curr_val > pandigital_prime):
            pandigital_prime = curr_val

    return pandigital_prime


def ProjectEuler_CodedTriangleNumbers_42() -> int:
    try:
        input_file = open(Path("input_files/0042_words.txt"), "r")
        words = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find list of first names")
        return -1

    for i in range(len(words)):
        curr_word = words[i]
        curr_score = 0

        for letter in curr_word:
            curr_score += ord(letter) - 64

        words[i] = curr_score

    max_word_val = max(words)

    triangle_nums = set()
    n = 1
    curr_triangle_num = 1

    while (curr_triangle_num <= max_word_val):
        triangle_nums.add(curr_triangle_num)
        n += 1
        curr_triangle_num = 0.5 * n * (n+1)

    num_triangle_words = 0
    for word_score in words:
        if word_score in triangle_nums:
            num_triangle_words += 1

    return num_triangle_words


def ProjectEuler_SubstringDivisibility_43() -> int:
    pandigital_sum = 0

    # Eliminated possible values for each digit based on criteria
    d1_digits = [1,2,3,4,6,7,8,9]
    d2_digits = [0,1,2,3,4,6,7,8,9]
    d3_digits = [0,1,2,3,4,6,7,8,9]
    d4_digits = [0,2,4,6,8]
    d5_digits = [3,9]
    d6 = 5
    d7_digits = [2,7]

    # Iterate over 3-digit numbers that are divisble by 17
    for d8_9_10 in range(102, 1000, 17):
        for d7 in d7_digits:
            # Check d6, d7, d8, d9, and d10 values are valid
            div_11 = (d6*100) + (d7*10) + (d8_9_10//100)
            div_13 = (d7*100) + (d8_9_10//10)

            if ((div_11 % 11) != 0) or ((div_13 % 13) != 0):
                continue

            for d5 in d5_digits:
                # Check d5 value is valid
                div_7 = (d5*100) + (d6*10) + d7
                if ((div_7 % 7) != 0):
                    continue

                # Iterate over even digits for d4
                for d4 in d4_digits:
                    for d3 in d3_digits:
                        # Check d3 value is valid
                        div_3 = d3+d4+d5
                        if ((div_3 % 3) != 0):
                            continue

                        for d1 in d1_digits:
                            for d2 in d2_digits:
                                # Calculate 10 digit val
                                d234 = (d2*100) + (d3*10) + d4
                                d567 = (d5*100) + (d6*10) + d7
                                val = (d1*1000000000) + (d234*1000000) + (d567*1000) + d8_9_10

                                # Ensure val is pandigital
                                if (pandigital(val)):
                                    pandigital_sum += val

    return pandigital_sum


def ProjectEuler_PentagonNumbers_44() -> int:
    # Calculate first 'n' pentagonal numbers
    n = 2500
    D = math.inf
    pent_nums = list()
    pent_nums_set = set()

    for i in range(1, n+1):
        curr_val = (i * ((3*i) - 1)) // 2
        pent_nums.append(curr_val)
        pent_nums_set.add(curr_val)

    for i in range(n):
        for j in range(i+1, n):
            val_i = pent_nums[i]
            val_j = pent_nums[j]
            if ((val_i + val_j) in pent_nums_set) and ((val_j - val_i) in pent_nums_set):
                if ((val_j - val_i) < D):
                    D = val_j - val_i

    return D


def ProjectEuler_TriPentHex_Numbers_45() -> int:
    n = 56000
    tri_nums = set()
    pent_nums = set()
    hex_nums = set()

    for i in range(144, n+1):
        tri_num = (i * (i+1)) / 2
        pent_num = (i * ((3*i) - 1)) / 2
        hex_num = i * ((2*i) - 1)

        tri_nums.add(tri_num)
        pent_nums.add(pent_num)
        hex_nums.add(hex_num)

    intersection = tri_nums & pent_nums & hex_nums

    if (len(intersection) != 1):
        print(f"Error: invalid number of solutions found")
        return -1

    result = int(intersection.pop())
    return result


def ProjectEuler_GoldbachOtherConjecture_46() -> int:
    try:
        input_file = open(Path("precomputed_primes/primes_1_million.txt"), "r")
        primes_set = json.load(input_file)
        primes_set = set(primes_set)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find list of all prime numbers less than one million")
        return -1

    n = 6000
    result = -1

    twice_squares = list()
    max_square = math.floor(math.sqrt(n))

    for i in range(1, max_square + 1):
        twice_squares.append((int(math.pow(i, 2)) * 2))

    for i in range(35, n, 2):
        if (check_prime(i)):
            continue

        for twice_square in twice_squares:
            complement = i - twice_square
            if complement in primes_set:
                break
        else:
            result = i
            break

    return result


def ProjectEuler_DistinctPrimesFactors_47() -> int:
    k = 4
    n = 135000
    num_pfactors = [0] * n
    valid_consec_nums = 0

    for i in range(2, n):
        # i is prime
        if (num_pfactors[i] == 0):
            valid_consec_nums = 0

            # Increment number of factors for all multiples of i up to n
            for j in range(i*2, n, i):
                num_pfactors[j] += 1

        # i is not prime but does not have 'k' distinct p_factors
        elif (num_pfactors[i] < k):
            valid_consec_nums = 0

        # i has 'k' distinct p_factors
        elif (num_pfactors[i] == k):
            valid_consec_nums += 1
            if (valid_consec_nums == k):
                return (i - k + 1)

    # No solution found, return -1
    return -1


def ProjectEuler_SelfPowers_48() -> int:
    n = 1000
    powers_sum = 0

    for i in range(1, n+1):
        powers_sum += i ** i

    solution = str(powers_sum)[-10:]
    return solution


def ProjectEuler_PrimePermutations_49() -> int:
    # Check all valid four digit numbers
    min_val = 1000
    max_val = 3340

    result = -1

    for i in range(min_val, max_val):
        if (i == 1487):
            continue

        j = i + 3330
        k = j + 3330

        if (check_prime(i) and check_prime(j) and check_prime(k)):
            if (set(str(i)) == set(str(j)) == set(str(k))):
                result = str(i) + str(j) + str(k)
                break

    return result


def ProjectEuler_ConsecutivePrimeSum_50() -> int:
    try:
        input_file = open(Path("precomputed_primes/primes_1_million.txt"), "r")
        primes = json.load(input_file)
        primes_set = set(primes)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find list of all prime numbers less than one million")
        return -1

    n = len(primes)
    max_val = primes[-1]
    max_consecutive_prime_sum = 0

    for k in range(21, 1000):
        prime_sum = 0
        for i in range(k):
            prime_sum += primes[i]

        if (prime_sum in primes_set):
            max_consecutive_prime_sum = prime_sum
            continue
        elif (prime_sum > max_val):
            break

        for i in range(k, n):
            prime_sum += primes[i]
            prime_sum -= primes[i-k]

            if (prime_sum in primes_set):
                max_consecutive_prime_sum = prime_sum
                break
            elif (prime_sum > max_val):
                break

    return max_consecutive_prime_sum


def main():
    sol_41 = ProjectEuler_PandigitalPrime_41()
    print(f"sol_41 = {sol_41}")

    sol_42 = ProjectEuler_CodedTriangleNumbers_42()
    print(f"sol_42 = {sol_42}")

    sol_43 = ProjectEuler_SubstringDivisibility_43()
    print(f"sol_43 = {sol_43}")

    sol_44 = ProjectEuler_PentagonNumbers_44()
    print(f"sol_44 = {sol_44}")

    sol_45 = ProjectEuler_TriPentHex_Numbers_45()
    print(f"sol_45 = {sol_45}")

    sol_46 = ProjectEuler_GoldbachOtherConjecture_46()
    print(f"sol_46 = {sol_46}")

    sol_47 = ProjectEuler_DistinctPrimesFactors_47()
    print(f"sol_47 = {sol_47}")

    sol_48 = ProjectEuler_SelfPowers_48()
    print(f"sol_48 = {sol_48}")

    sol_49 = ProjectEuler_PrimePermutations_49()
    print(f"sol_49 = {sol_49}")

    sol_50 = ProjectEuler_ConsecutivePrimeSum_50()
    print(f"sol_50 = {sol_50}")


if __name__ == "__main__":
    main()
