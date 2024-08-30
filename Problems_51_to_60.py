# Keshav Pandey
from collections import Counter
import json
import math
from pathlib import Path
from ProjectEuler_Helpers import concatenate_integers
import re

def check_prime_family(val: list, indices: list, memo: set, prime_family_size: int) -> bool:
    digits = ['9','8','7','6','5','4','3','2','1','0']
    allowed_fails = 10 - prime_family_size

    for digit in digits:
        for index in indices:
            val[index] = digit

        if (tuple(val) not in memo):
            allowed_fails -=1

            if (allowed_fails < 0):
                return False

    return True


class PokerHand():
    # Store hands and the ranking of each hand in a dict
    poker_hands = {
        "royal_flush": 9,
        "straight_flush": 8,
        "four_of_kind": 7,
        "full_house": 6,
        "flush": 5,
        "straight": 4,
        "three_of_kind": 3,
        "two_pair": 2,
        "one_pair": 1,
        "none": 0
    }

    def __init__(self, hand: list):
        self.hand = hand
        self.hand_rank = -1

        self.card_values = Counter()
        self.card_suits = Counter()
        self.tie_breaker_values = list()

        # Call helper function to initialize all above object variables
        self.set_hand_variables()

    # Sets rank of current hand. A larger rank value = better hand
    def set_hand_variables(self) -> bool:
        # Get frequencies of each card suit and card value in hand
        for card in self.hand:
            self.card_suits[card[1]] += 1

            if (card[0] == 'T'):
                self.card_values[10] += 1
            elif (card[0] == 'J'):
                self.card_values[11] += 1
            elif (card[0] == 'Q'):
                self.card_values[12] += 1
            elif (card[0] == 'K'):
                self.card_values[13] += 1
            elif (card[0] == 'A'):
                self.card_values[14] += 1
            else:
                self.card_values[int(card[0])] += 1

        min_card_val = min(self.card_values.keys())
        max_card_val = max(self.card_values.keys())

        rank_value = -1
        non_rank_value = -1

        if (len(self.hand) != 5):
            print("Error: invalid number of cards in hand")
            return False

        # Check royal_flush and straight_flush:
        if (len(self.card_suits) == 1) and (len(self.card_values) == 5):
            if (min_card_val == 10) and (max_card_val == 14):
                self.hand_rank = self.poker_hands["royal_flush"]
                return True

            elif ((min_card_val + 4) == max_card_val):
                self.hand_rank = self.poker_hands["straight_flush"]
                self.tie_breaker_values.append(max_card_val)
                return True

        # Check four_of_kind
        if (len(self.card_values) == 2):
            for value, freq in self.card_values.items():
                if (freq == 4):
                    rank_value = value
                elif (freq == 1):
                    non_rank_value = value
                else:
                    break
            else:
                self.hand_rank = self.poker_hands["four_of_kind"]
                self.tie_breaker_values.append(rank_value)
                self.tie_breaker_values.append(non_rank_value)
                return True

        # Check full_house
        if (len(self.card_values) == 2):
            for value, freq in self.card_values.items():
                if (freq == 3):
                    rank_value = value
                elif (freq == 2):
                    non_rank_value = value
                else:
                    print("Logical Error. Failed to determine hand rank.")
                    return False
            self.hand_rank = self.poker_hands["full_house"]
            self.tie_breaker_values.append(rank_value)
            self.tie_breaker_values.append(non_rank_value)
            return True

        # Check flush
        if (len(self.card_suits) == 1):
            self.hand_rank = self.poker_hands["flush"]
            self.tie_breaker_values += list(self.card_values.keys())
            self.tie_breaker_values.sort(reverse=True)
            return True

        # Check straight
        if (len(self.card_values) == 5) and ((min_card_val + 4) == max_card_val):
            self.hand_rank = self.poker_hands["straight"]
            self.tie_breaker_values.append(max_card_val)
            return True

        # Check three_of_kind
        non_rank_values = list()
        if (3 in self.card_values.values()):
            for value, freq in self.card_values.items():
                if (freq == 3):
                    rank_value = value
                elif (freq == 1):
                    non_rank_values.append(value)
                else:
                    print("Logical Error. Failed to determine hand rank.")
                    return False
            self.hand_rank = self.poker_hands["three_of_kind"]
            self.tie_breaker_values.append(rank_value)
            non_rank_values.sort(reverse=True)
            self.tie_breaker_values += non_rank_values
            return True

        # Check two_pair, one_pair
        num_pairs = 0
        rank_values = list()
        non_rank_values = list()

        for value, freq in self.card_values.items():
            if (freq == 2):
                rank_values.append(value)
                num_pairs += 1
            elif (freq == 1):
                non_rank_values.append(value)

        rank_values.sort(reverse=True)
        non_rank_values.sort(reverse=True)
        self.tie_breaker_values += (rank_values + non_rank_values)

        if (num_pairs == 2):
            self.hand_rank = self.poker_hands["two_pair"]
            return True
        elif (num_pairs == 1):
            self.hand_rank = self.poker_hands["one_pair"]
            return True

        # No special hands found. Return "none"
        self.hand_rank = self.poker_hands["none"]
        return True

    def __lt__(self, other):
        if (self.hand_rank != other.hand_rank):
            return self.hand_rank < other.hand_rank

        for i in range(len(self.tie_breaker_values)):
            if (self.tie_breaker_values[i] != other.tie_breaker_values[i]):
                return self.tie_breaker_values[i] < other.tie_breaker_values[i]

        return False

    def __gt__(self, other):
        if (self.hand_rank != other.hand_rank):
            return self.hand_rank > other.hand_rank

        for i in range(len(self.tie_breaker_values)):
            if (self.tie_breaker_values[i] != other.tie_breaker_values[i]):
                return self.tie_breaker_values[i] > other.tie_breaker_values[i]

        return False


    def __eq__(self, other):
        if (self.hand_rank != other.hand_rank):
            return False

        for i in range(len(self.tie_breaker_values)):
            if (self.tie_breaker_values[i] != other.tie_breaker_values[i]):
                return False

        return True


# Helper function that determines if input parameter is prime
# This prime number checker uses the Miller–Rabin primality test
def check_likely_prime(num: int) -> bool:
    # Smallest and only even prime number is 2
    if (num < 2):
        return False
    elif (num == 2):
        return True
    elif ((num % 2) == 0):
        return False
    elif (num >= 4759123141):
        print(f"Warning: primality check not valid for {num}. Must increase witness set.")

    # If (num < 4,759,123,141): only need to test witnesses a = [2, 7, 61]
    a_vals = [2,7,61]

    # Calculate constants 's' and 'd'
    d = (num-1) // 2
    s = 1

    while ((d % 2) == 0):
        d //= 2
        s += 1

    # Test each 'a' value against input num
    for a in a_vals:
        # Only test 'a' values in range [2, n-2]
        if (a >= (num-1)):
            break

        # Initialize: x = (a^d) % num
        x = pow(a, d, mod=num)

        # Check congruence relationship 's' times
        for _ in range(s):
            y = pow(x, 2, mod=num)
            if (y == 1) and (x != 1) and (x != (num-1)):
                return False
            x = y

        if (y != 1):
            return False

    return True


def check_prime_pair(x: int, y: int, memo: set) -> bool:
    # Check concatentating x and y
    val1 = concatenate_integers(x, y)
    if val1 not in memo:
        return False

    # Check concatentating y and x
    val2 = concatenate_integers(y, x)
    if val2 not in memo:
        return False

    return True


def ProjectEuler_PrimeDigitReplacements_51() -> int:
    try:
        input_file = open(Path("precomputed_primes/primes_1_million.txt"), "r")
        primes_list = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find list of prime numbers")
        return -1

    # Define constants
    n = len(primes_list)
    prime_family_size = 8

    # It is advantageous to work with lists to easily modify and iterate over digits
    for i in range(n):
        primes_list[i] = list(str(primes_list[i]))

    primes_set = set()
    for i in range(n):
        primes_set.add(tuple(primes_list[i]))

    # Try to find solution by swapping exactly three digits
    for i in range(n):
        curr_prime = primes_list[i]
        num_swap_digits = len(curr_prime)-1

        for idx1 in range(num_swap_digits):
            digit1 = curr_prime[idx1]

            for idx2 in range(idx1+1, num_swap_digits):
                digit2 = curr_prime[idx2]
                if (digit1 != digit2):
                    continue

                for idx3 in range(idx2+1, num_swap_digits):
                    digit3 = curr_prime[idx3]
                    if (digit2 != digit3):
                        continue

                    if (check_prime_family(curr_prime.copy(), [idx1, idx2, idx3], primes_set, prime_family_size)):
                        result = int("".join(curr_prime))
                        return result

    return -1


def ProjectEuler_PermutedMultiples_52() -> int:
    n = 150000

    for i in range(1, n):
        if (len(str(i)) != len(str(i*6))):
            continue

        if (set(str(i)) != set(str(i*2))) or (set(str(i)) != set(str(i*3))) or \
            (set(str(i)) != set(str(i*4))) or (set(str(i)) != set(str(i*5))) or \
            set(str(i)) != set(str(i*6)):
            continue

        result = i
        break

    return result


def ProjectEuler_CombinatoricSelections_53() -> int:
    max_n = 100
    threshold = 1000000
    result = 0

    for n in range(1, max_n + 1):
        for r in range(1, n):
            if (math.comb(n, r) > threshold):
                result += 1

    return result


def ProjectEuler_PokerHands_54() -> int:
    try:
        input_file_path = Path("input_files/0054_poker.txt")
        input_file = open(input_file_path, "r")
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find file containing 1000 poker hands")
        return -1

    num_player1_wins = 0
    num_player2_wins = 0
    num_ties = 0

    with open(input_file_path, "r") as input:
        for line in input:
            line = line.strip().split(" ")
            player1_hand = line[:5]
            player2_hand = line[5:]

            player1 = PokerHand(player1_hand)
            player2 = PokerHand(player2_hand)

            if (player1 > player2):
                num_player1_wins += 1
            elif (player1 < player2):
                num_player2_wins += 1
            else:
                num_ties += 1

    return num_player1_wins


def ProjectEuler_LychrelNumbers_55() -> int:
    n = 10000
    max_iterations = 50
    total_lychrel_nums = 0

    for i in range(1, n):
        curr_val = i
        for _ in range(max_iterations):
            palindrome = int(str(curr_val)[::-1])
            new_val = curr_val + palindrome
            if (str(new_val) == str(new_val)[::-1]):
                break
            else:
                curr_val = new_val
        else:
            total_lychrel_nums += 1

    return total_lychrel_nums


def ProjectEuler_PowerfulDigitSum_56() -> int:
    n = 100
    max_digit_sum = 0

    for a in range(1, n):
        for b in range(1, n):
            power = a ** b
            curr_sum = 0

            while (power > 0):
                curr_digit = power % 10
                power = power // 10
                curr_sum += curr_digit

            if (curr_sum > max_digit_sum):
                max_digit_sum = curr_sum

    return max_digit_sum


def ProjectEuler_SquareRootConvergents_57() -> int:
    n = 1000

    # Initialize variables with values at iteration zero
    num = 1
    denom = 1
    num_more_digits_count = 0

    for _ in range(n):
        num += denom
        num, denom = denom, num
        num += denom

        if len(str(num)) > len(str(denom)):
            num_more_digits_count += 1

    return num_more_digits_count


def ProjectEuler_SpiralPrimes_58() -> int:
    # Initialize constants:
    num_sides = 4
    threshold = 0.10

    # Initialize variables:
    curr_val = 1
    curr_side_length = 1

    num_primes = 0
    num_diag_vals = 1

    ratio = 1

    while (ratio >= threshold):
        curr_side_length += 2

        # Iterate and check primality of first three diagonal elements only
        # Fourth diagonal element will always be square (9, 25, 49, etc.) -> not prime
        for _ in range(num_sides-1):
            curr_val += curr_side_length - 1
            if check_likely_prime(curr_val):
                num_primes += 1

        # Skip over fourth diagonal value
        curr_val += curr_side_length - 1

        # Calculate new ratio for current sidelength
        num_diag_vals += 4
        ratio = num_primes / num_diag_vals

    return curr_side_length


def ProjectEuler_XORDecryption_59() -> int:
    try:
        input_file = open(Path("input_files/0059_cipher.txt"), "r")
        encrypted_input = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find encrypted ASCII codes")
        return -1

    n = len(encrypted_input)
    for i in range(n):
        encrypted_input[i] = int(encrypted_input[i])

    key_min = ord("a")
    key_max = ord("z")
    regex_expr = "^[a-zA-Z0-9\s.,/+:;()'\"\[\]]*$"
    result = -1

    for i in range(key_min, key_max+1):
        if (re.search(regex_expr, chr(encrypted_input[0] ^ i)) == None):
            continue

        for j in range(key_min, key_max+1):
            if (re.search(regex_expr, chr(encrypted_input[1] ^ j)) == None):
                continue

            for k in range(key_min, key_max+1):
                encryption_key = [i, j, k]
                key_idx = 0
                decrypted_input = []

                for curr_val in encrypted_input:
                    curr_key = encryption_key[key_idx % 3]
                    curr_decrypt = curr_val ^ curr_key

                    if (re.search(regex_expr, chr(curr_decrypt)) == None):
                        break
                    else:
                        decrypted_input.append(curr_decrypt)
                        key_idx += 1
                else:
                    result = sum(decrypted_input)
                    return result

    return result


def ProjectEuler_PrimePairSets_60() -> int:
    try:
        input_file = open(Path("precomputed_primes/primes_100_million.txt"), "r")
        prime_check_set = set(json.load(input_file))
        input_file.close()

        input_file = open(Path("precomputed_primes/primes_10_thousand.txt"), "r")
        prime_list = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find lists of prime numbers")
        return -1

    # Initialize variables
    n = len(prime_list)
    minimum_sum = math.inf

    # Iterate over all pairs of primes and see if they also concatenate into primes
    # Save result in boolean matrix. Assume we always check [i,j] pairs where i < j
    isConcatPrime = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if check_prime_pair(prime_list[i], prime_list[j], prime_check_set):
                isConcatPrime[i][j] = True

    # Get 1st prime number in sequence
    for a in range(n-4):
        # Get 2nd prime number in sequence
        for b in range(a+1, n-3):
            if not (isConcatPrime[a][b]):
                continue

            # Get 3rd prime number in sequence
            for c in range(b+1, n-2):
                sum_c = prime_list[a] + prime_list[b] + prime_list[c]
                if (sum_c >= minimum_sum):
                    break

                if not (isConcatPrime[a][c] and isConcatPrime[b][c]):
                    continue

                # Get 4th prime number in sequence
                for d in range(c+1, n-1):
                    sum_d = sum_c + prime_list[d]
                    if (sum_d >= minimum_sum):
                        break

                    if not (isConcatPrime[a][d] and isConcatPrime[b][d] and isConcatPrime[c][d]):
                        continue

                    # Get 5th prime number in sequence
                    for e in range(d+1, n):
                        sum_e = sum_d + prime_list[e]
                        if (sum_e >= minimum_sum):
                            break

                        if (isConcatPrime[a][e] and isConcatPrime[b][e] and isConcatPrime[c][e] and isConcatPrime[d][e]):
                            minimum_sum = sum_e

    return minimum_sum


def main():
    sol_51 = ProjectEuler_PrimeDigitReplacements_51()
    print(f"sol_51 = {sol_51}")

    sol_52 = ProjectEuler_PermutedMultiples_52()
    print(f"sol_52 = {sol_52}")

    sol_53 = ProjectEuler_CombinatoricSelections_53()
    print(f"sol_53 = {sol_53}")

    sol_54 = ProjectEuler_PokerHands_54()
    print(f"sol_54 = {sol_54}")

    sol_55 = ProjectEuler_LychrelNumbers_55()
    print(f"sol_55 = {sol_55}")

    sol_56 = ProjectEuler_PowerfulDigitSum_56()
    print(f"sol_56 = {sol_56}")

    sol_57 = ProjectEuler_SquareRootConvergents_57()
    print(f"sol_57 = {sol_57}")

    sol_58 = ProjectEuler_SpiralPrimes_58()
    print(f"sol_58 = {sol_58}")

    sol_59 = ProjectEuler_XORDecryption_59()
    print(f"sol_59 = {sol_59}")

    sol_60 = ProjectEuler_PrimePairSets_60()
    print(f"sol_60 = {sol_60}")


if __name__ == "__main__":
    main()
