# Keshav Pandey
import heapq
from itertools import combinations
import json
import math
from pathlib import Path
import random

def roll_dice(num_rolls: int, max_dice_val: int) -> int:
    result = 0
    rolls = set()
    double = False

    for _ in range(num_rolls):
        roll = random.randint(1, max_dice_val)
        rolls.add(roll)
        result += roll

    if (len(rolls) != num_rolls):
        double = True

    return (result, double)


# TODO: get better name
# Get number of ways we can sum to 'n' using only two integers 'x' and 'y'
# Such that: 1 <= x <= y <= max_addend
def get_num_combinations(n: int, max_addend: int) -> int:
    curr_addend = min(max_addend, n-1)
    result = 0

    while ((n-curr_addend) <= curr_addend):
        curr_addend -= 1
        result += 1

    return result

    # for i in range(1, n+1):
    #     memo[i] = memo[i-1]
    #     if (i-curr_addend) >= curr_addend:
    #         memo[i] += 1
    #         max_addend += 1

    # # for i in range(1, max_n+1):
    # #     memo[i] = memo[i-1]
    # #     if (i-max_addend) >= max_addend:
    # #         memo[i] += 1
    # #         max_addend += 1

    # return memo


def get_prime_factors_list(val: int, prime_list: list, memo: list) -> list:
    prime_factors = list()
    for prime in prime_list:
        if ((val % prime) == 0):
            val //= prime
            prime_factors += memo[val] if memo[val] else [val]
            prime_factors.append(prime)
            return prime_factors

    return -1


# Return list of ways to multiply to 'num' with 2 values (f1*f2)
def getcombo(num: int, pfactors: list, min_r: int, multipliers: int, memo) -> list:
    curr_pfactors = pfactors[num]
    num_pfactors = len(curr_pfactors)
    product_combos = list()

    if (multipliers == 2):
        for r in range(min_r, (num_pfactors//2)+1):
            multiplicands = list(combinations(curr_pfactors, r))

            for multiplicand in multiplicands:
                f1 = 1
                for factor in multiplicand:
                    f1 *= factor
                product_combos.append([f1, num//f1])
    # elif (multipliers == 3):
    else:
        # for r in range(min_r, (num_pfactors//3)+1):
        for r in range(min_r, (num_pfactors//multipliers)+1):
            multiplicands1 = list(combinations(curr_pfactors, r))

            for mult1_combo in multiplicands1:
                mult1 = 1
                for mult1_factor in mult1_combo:
                    mult1 *= mult1_factor

                new_num = num // mult1

                multiplicands2 = list(memo[new_num][multipliers-1])

                # if (memo.get((new_num, r, multipliers-1), None) != None):
                #     multiplicands2 = memo.get((new_num, r, multipliers-1))
                # else:
                #     multiplicands2 = getcombo(new_num, pfactors, r, multipliers-1, memo)
                #     memo[(new_num, r, multipliers-1)] = multiplicands2

                # multiplicands2 = getcombo(new_num, pfactors, r, multipliers-1, memo)

                for combo in multiplicands2:
                    product_combos.append([mult1]+[combo])
    # else:
    #     print(f"error, too many groups: num={num}, groups={multipliers}, p_factors={curr_pfactors}")
    #     import sys; sys.exit();

    return product_combos


def roman_to_int(map: dict, numeral: str) -> int:
    result = 0
    prev_digit = 0

    for i in range(len(numeral)-1, -1, -1):
        digit = map[numeral[i]]
        if (digit < prev_digit):
            result -= digit
        else:
            result += digit
        prev_digit = digit

    return result


def int_to_roman(num: int) -> str:
    result = ""
    roman_convrt = [('M', 1000, "CM", 900),
                    ('D',  500, "CD", 400),
                    ('C',  100, "XC",  90),
                    ('L',   50, "XL",  40),
                    ('X',   10, "IX",   9),
                    ('V',    5, "IV",   4),
                    ('I',    1,   "", math.inf)]

    for (symb, val, sub_symb, sub_val) in roman_convrt:
        while ((num // val) > 0):
            result += symb
            num -= val

        if (num >= sub_val):
            result +=sub_symb
            num -= sub_val

    return result


def ProjectEuler_PathSum_TwoWays_81() -> int:
    try:
        input_file = open(Path("input_files/0081_matrix.txt"), "r")
        matrix = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find input matrix")
        return -1

    # Initialize dynamic programming matrix
    n = len(matrix)
    dp = [-1] * n

    # Set base cases
    dp[0] = matrix[0][0]
    for i in range(1, n):
        dp[i] = matrix[0][i] + dp[i-1]

    # Iterate through input matrix row by row to find min path to each node
    for i in range(1, n):
        dp[0] += matrix[i][0]
        for j in range(1, n):
            dp[j] = min(dp[j], dp[j-1]) + matrix[i][j]

    return dp[n-1]


def ProjectEuler_PathSum_ThreeWays_82() -> int:
    try:
        input_file = open(Path("input_files/0082_matrix.txt"), "r")
        matrix = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find input matrix")
        return -1

    # Initialize dynamic programming lists
    n = len(matrix)
    prev_col = list()
    curr_col = [0] * n

    # Set base cases
    for i in range(n):
        curr_col[i] = matrix[i][0]

    # We iterate column by column instead of row by row:
    for i in range(1, n):
        prev_col = list(curr_col)
        curr_col = [0] * n

        for j in range(0, n):
            # Try going right:
            curr_col[j] = prev_col[j]

            # Try going down:
            dist_cost = 0
            for k in range(j+1, n):
                dist_cost += matrix[k-1][i-1]
                curr_col[j] = min(curr_col[j], prev_col[k] + dist_cost)

            # Try going up:
            dist_cost = 0
            for k in range(j-1, -1, -1):
                dist_cost += matrix[k+1][i-1]
                curr_col[j] = min(curr_col[j], prev_col[k] + dist_cost)

            # Add curr cost
            curr_col[j] += matrix[j][i]

    # Find which node in last column of matrix has cheapest path cost
    min_cost = min(curr_col)
    return min_cost


def ProjectEuler_PathSum_FourWays_83() -> int:
    try:
        input_file = open(Path("input_files/0083_matrix.txt"), "r")
        matrix = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find input matrix")
        return -1

    n = len(matrix)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    p_queue = []
    visited = set()
    shortest_paths = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            shortest_paths[i][j] = math.inf
    shortest_paths[0][0] = matrix[0][0]

    # Enqueue start node info: (distance to itself, i location, j location)
    heapq.heappush(p_queue, (matrix[0][0], 0, 0))

    while (len(p_queue) > 0):
        curr_dist, curr_i, curr_j = heapq.heappop(p_queue)
        visited.add((curr_i, curr_j))

        for i_diff, j_diff in directions:
            new_i = curr_i + i_diff
            new_j = curr_j + j_diff

            if (new_i < 0) or (new_j < 0) or (new_i >= n) or (new_j >= n) or ((new_i, new_j) in visited):
                continue

            new_dist = curr_dist + matrix[new_i][new_j]

            if (new_dist < shortest_paths[new_i][new_j]):
                shortest_paths[new_i][new_j] = new_dist
                heapq.heappush(p_queue, (new_dist, new_i, new_j))

    return shortest_paths[n-1][n-1]


def ProjectEuler_MonopolyOdds_84() -> int:
    board = ["GO", "A1", "CC1", "A2", "T1", "R1", "B1", "CH1",
             "B2", "B3", "JAIL", "C1", "U1", "C2", "C3", "R2",
             "D1", "CC2", "D2", "D3", "FP", "E1", "CH2", "E2",
             "E3", "R3", "F1", "F2", "U2", "F3", "G2J", "G1",
             "G2", "CC3", "G3", "R4", "CH3", "H1", "T2", "H2"]

    monopoly_map = {"GO": 0,    "A1": 1,   "CC1": 2,  "A2": 3,   "T1": 4,
                    "R1": 5,    "B1": 6,   "CH1": 7,  "B2": 8,   "B3": 9,
                    "JAIL": 10, "C1": 11,  "U1": 12,  "C2": 13,  "C3": 14,
                    "R2": 15,   "D1": 16,  "CC2": 17, "D2": 18,  "D3": 19,
                    "FP": 20,   "E1": 21,  "CH2": 22, "E2": 23,  "E3": 24,
                    "R3": 25,   "F1": 26,  "F2": 27,  "U2": 28,  "F3": 29,
                    "G2J": 30,  "G1": 31,  "G2": 32,  "CC3": 33, "G3": 34,
                    "R4": 35,   "CH3": 36, "H1": 37,  "T2": 38,  "H2": 39}

    community_chest = ["GO", "JAIL", None, None, None, None, None, None,
                       None,  None,  None, None, None, None, None, None]
    cc_num_cards = len(community_chest)

    chance = ["GO", "JAIL", "C1", "E3", "H2", "R1", "NEXT_R", "NEXT_R",
              "NEXT_U", "BACK_3", None, None, None, None, None, None]
    ch_num_cards = len(chance)

    # Initialize constants
    num_sqrs = len(board)
    num_dice = 2

    # Initialize parameters
    n_turns = 250000
    dice_size = 4
    num_visits = [0] * num_sqrs

    # Initialize variables to play game
    sqr_idx = 0
    cc_card_idx = 0
    ch_card_idx = 0
    consec_doubles = 0

    for _ in range(n_turns):
        increment_val, double = roll_dice(num_dice, dice_size)

        # Check for three consecutive doubles -> send to jail if occurs
        if (double):
            consec_doubles += 1

            if (consec_doubles >= 3):
                sqr_idx = monopoly_map["JAIL"]
                num_visits[sqr_idx] += 1
                consec_doubles = 0
                continue
        else:
            consec_doubles = 0

        # Move to next square based on dice roll
        sqr_idx = (sqr_idx + increment_val) % num_sqrs

        # Check if landed on a special square
        if (board[sqr_idx] == "G2J"):
            sqr_idx = monopoly_map["JAIL"]

        elif (board[sqr_idx] == "CC1") or (board[sqr_idx] == "CC2") or (board[sqr_idx] == "CC3"):
            cc_card = community_chest[cc_card_idx]
            if (cc_card != None):
                sqr_idx = monopoly_map[cc_card]
            cc_card_idx = (cc_card_idx + 1) % cc_num_cards

        elif (board[sqr_idx] == "CH1") or (board[sqr_idx] == "CH2") or (board[sqr_idx] == "CH3"):
            ch_card = chance[ch_card_idx]
            if (ch_card != None):
                if (ch_card == "NEXT_R"):
                    if (board[sqr_idx] == "CH1"): sqr_idx = monopoly_map["R2"]
                    elif (board[sqr_idx] == "CH2"): sqr_idx = monopoly_map["R3"]
                    elif (board[sqr_idx] == "CH3"): sqr_idx = monopoly_map["R1"]
                elif (ch_card == "NEXT_U"):
                    if (board[sqr_idx] == "CH1"): sqr_idx = monopoly_map["U1"]
                    elif (board[sqr_idx] == "CH2"): sqr_idx = monopoly_map["U2"]
                    elif (board[sqr_idx] == "CH3"): sqr_idx = monopoly_map["U1"]
                elif (ch_card == "BACK_3"):
                    sqr_idx -= 3
                    if (sqr_idx < 0): sqr_idx += num_sqrs
                else:
                    sqr_idx = monopoly_map[ch_card]
            ch_card_idx = (ch_card_idx + 1) % ch_num_cards

        # Increment num_visits counter for final square of this turn
        num_visits[sqr_idx] += 1

    # Get resulting 6-digit modal string
    result = ""
    for _ in range(3):
        max_val = -1
        max_idx = -1

        for i in range(num_sqrs):
            if (num_visits[i] > max_val):
                max_val = num_visits[i];
                max_idx = i

        if (max_idx < 10):
            result += "0"

        result += str(max_idx)
        num_visits[max_idx] = -1

    return result


def ProjectEuler_CountingRectangles_85() -> int:
    n = 2000000
    best_area = -1
    best_diff = math.inf
    choose_twos = list()

    for i in range(2, 100):
        val = math.comb(i, 2)
        choose_twos.append((i-1, val))

    for x_idx in range(len(choose_twos)):
        x_dim, x_val = choose_twos[x_idx]
        for y_idx in range(x_idx, len(choose_twos)):
            y_dim, y_val = choose_twos[y_idx]
            curr_rects = x_val * y_val
            curr_diff = abs(curr_rects-n)

            if (curr_diff < best_diff):
                best_diff = curr_diff
                best_area = x_dim * y_dim

    return best_area


# TODO: optimize using Pythagorean triangle DP method
def ProjectEuler_CuboidRoute_86() -> int:
    M_max = 2000
    n = 1000000
    valid_routes = 0

    # Pre-calculate and store all needed squares to avoid re-calculating each time
    sqrs_len = (M_max*2) + 1
    sqrs = [None] * sqrs_len
    for i in range(1, sqrs_len):
        sqrs[i] = (int(math.pow(i, 2)))

    base = 0
    while (valid_routes < n):
        base += 1

        # Iterate over all possible triangle heights such that base remains longest side length
        for height in range(2, (2*base)+1):
            # Min path will always have longest side length as the triangle base
            min_path = math.sqrt(sqrs[base] + sqrs[height])

            # If current base/height combination is valid solution, account for all prism
            # side lengths (a,b,c) such that (a=base, b+c=height)
            if (min_path == int(min_path)):
                valid_routes += get_num_combinations(height, base)

    return base


def ProjectEuler_PrimePowerTriples_87() -> int:
    try:
        input_file = open(Path("precomputed_primes/primes_10_thousand.txt"), "r")
        primes_list = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find list of prime numbers")
        return -1

    n = 50000000
    unique_sums = set()
    result = 0

    squares = list()
    for prime in primes_list:
        curr_sqr = math.pow(prime, 2)
        if (curr_sqr > n): break
        squares.append(curr_sqr)

    cubes = list()
    for prime in primes_list:
        curr_cube = math.pow(prime, 3)
        if (curr_cube > n): break
        cubes.append(curr_cube)

    fourths = list()
    for prime in primes_list:
        curr_four = math.pow(prime, 4)
        if (curr_four > n): break
        fourths.append(curr_four)

    for a in fourths:
        for b in cubes:
            if (a+b) >= n:
                break
            for c in squares:
                num = a + b + c
                if (num >= n):
                    break
                unique_sums.add(num)

    result = len(unique_sums)
    return result


def ProjectEuler_ProductSum_Numbers_88() -> int:
    try:
        input_file = open(Path("precomputed_primes/primes_100_thousand.txt"), "r")
        primes_list = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find list of prime numbers")
        return -1

    # Assume all min product-sums for k <= k_max are < prodsum_max
    prodsum_max = 12500
    primes_set = set(primes_list)
    pfactors_list = [None] * prodsum_max

    for num in range(2, prodsum_max):
        if (num in primes_set):
            continue

        num_pfactors = get_prime_factors_list(num, primes_list, pfactors_list)
        pfactors_list[num] = num_pfactors

    # print(pfactors_list)
    # print(pfactors_list[soln_max-1], soln_max-1)
    # print(pfactors_list[283], 283)

    max_pfactors = 14
    memo_grid = [[None] * max_pfactors for _ in range(prodsum_max)]

    for num in range(2, prodsum_max):
        curr_pfactors = pfactors_list[num]
        if (curr_pfactors == None):
            # print(num)
            continue

        max_vars = len(curr_pfactors)

        for num_vars in range(2, max_vars+1):
            curr_combos = getcombo(num, pfactors_list, 1, num_vars, memo_grid)
            memo_grid[num][num_vars] = set()

            for combo in curr_combos:
                curr_sum = sum(combo)
                memo_grid[num][num_vars].add(curr_sum)
            # print(f"num={num}, num_vars={num_vars}, curr_combos={curr_combos}, curr_sums={memo_grid[num][num_vars]}")

    # print(f"******************************************")
    # for i in range(len(memo_grid)):
    #     print(f"i={i} ; memo[i]={memo_grid[i]}")


    k_max = 12000
    prod_sums = set()
    total_min_prod_sum = 0

    for k in range(2, k_max+1):
        # print(f"****k = {k}")
        searching = True
        num_ones = k-2
        num_vars = 2
        i = k+1

        while (i < prodsum_max) and (searching):
            # Solution cannot be prime number (probs), skip it
            if (pfactors_list[i] == None):
                # print(f"skip prime: {i}")
                i += 1
                continue

            # Solution does not have enough pfactors, skip it
            curr_pfactors = pfactors_list[i]
            if (num_vars > len(curr_pfactors)):
                # print(f"skip too few pfactors: {i}")
                i += 1
                continue

            # print(f"k={k},i={i},p_factors={curr_pfactors}")

            # # Start by testing with only 2 pfactors, and increment gradually if no soln found
            while (num_vars <= len(curr_pfactors)) and (searching):
                target = i-num_ones
                if (target in memo_grid[i][num_vars]):
                    # print(f"yayyy for k={k}, min prod-sum = {i}, {num_vars}")
                    searching = False
                    if (i not in prod_sums):
                        total_min_prod_sum += i
                        prod_sums.add(i)
                else:
                    num_vars += 1
                    num_ones -= 1

            num_ones = k-2
            num_vars = 2
            i += 1


            # # Start by testing with only 2 pfactors, and increment gradually if no soln found
            # while (num_vars <= len(curr_pfactors)) and (searching):
            #     if (memo.get((i, 1, num_vars), None) != None):
            #         test_combos = memo.get((i, 1, num_vars))
            #     else:
            #         test_combos = getcombo(i, pfactors_list, 1, num_vars, memo)
            #         memo[(i, 1, num_vars)] = test_combos

            #     for combo in test_combos:
            #         curr_sum = num_ones
            #         for curr_val in combo:
            #             curr_sum += curr_val

            #         if (curr_sum == i):
            #             print(f"yayyy for k={k}, min prod-sum = {i}, {combo}")
            #             searching = False
            #             if (i not in prod_sums):
            #                 total_min_prod_sum += i
            #                 prod_sums.add(i)
            #             break

            #     num_vars += 1
            #     num_ones -= 1

            # num_ones = k-2
            # num_vars = 2
            # i += 1


    return total_min_prod_sum


def ProjectEuler_RomanNumerals_89() -> int:
    try:
        input_file = open(Path("input_files/0089_roman.txt"), "r")
        roman_numerals = input_file.readlines()
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find input roman numerals")
        return -1

    roman_map = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
    num_saved_chars = 0

    for i in range(len(roman_numerals)):
        input_numeral = roman_numerals[i].strip()
        int_val = roman_to_int(roman_map, input_numeral)
        roman_val = int_to_roman(int_val)
        num_saved_chars += len(input_numeral) - len(roman_val)

    return num_saved_chars


def ProjectEuler_CubeDigitPairs_90() -> int:
    combos = combinations([0,1,2,3,4,5,6,7,8,9], 6)
    valid_cube_combos = list()

    for combo in combos:
        if (2 in combo) or (5 in combo):
            bit_combo = 0
            for val in combo:
                bit_combo |= (1 << val)
            valid_cube_combos.append(bit_combo)

    # Constants:
    zero_mask   = 0b1
    one_mask    = 0b10
    two_mask    = 0b100
    three_mask  = 0b1000
    four_mask   = 0b10000
    five_mask   = 0b100000
    six_mask    = 0b1000000
    eight_mask  = 0b100000000
    nine_mask   = 0b1000000000

    # Initialize variables for finding solution
    n = len(valid_cube_combos)
    counter = 0

    for i in range(n):
        cube1 = valid_cube_combos[i]
        cube1_zero = (cube1 & zero_mask) == zero_mask

        if (not cube1_zero):
            break

        for j in range(i, n):
            cube2 = valid_cube_combos[j]
            cube2_zero = (cube2 & zero_mask) == zero_mask

            # Check '01', '04', and '09' case
            cube1_one = (cube1 & one_mask) == one_mask
            cube1_four = (cube1 & four_mask) == four_mask
            cube1_sixnine = ((cube1 & six_mask) == six_mask) or ((cube1 & nine_mask) == nine_mask)

            cube2_one = (cube2 & one_mask) == one_mask
            cube2_four = (cube2 & four_mask) == four_mask
            cube2_sixnine = ((cube2 & six_mask) == six_mask) or ((cube2 & nine_mask) == nine_mask)

            # Both cubes have a '0'
            if (cube2_zero):
                if not (cube1_one or cube2_one):
                    continue
                if not (cube1_four or cube2_four):
                    continue
                if not (cube1_sixnine or cube2_sixnine):
                    continue
            # Only cube1 has a '0'
            else:
                if not (cube2_one and cube2_four and cube2_sixnine):
                    continue

            # Check '25' case
            cube1_two = (cube1 & two_mask) == two_mask
            cube1_five = (cube1 & five_mask) == five_mask

            cube2_two = (cube2 & two_mask) == two_mask
            cube2_five = (cube2 & five_mask) == five_mask

            if not ((cube1_two and cube2_five) or (cube2_two and cube1_five)):
                continue

            # Check '16' case
            if not ((cube1_one and cube2_sixnine) or (cube2_one and cube1_sixnine)):
                continue

            # Check '36' case
            cube1_three = (cube1 & three_mask) == three_mask
            cube2_three = (cube2 & three_mask) == three_mask

            if not ((cube1_three and cube2_sixnine) or (cube2_three and cube1_sixnine)):
                continue

            # Check '49' and '64' case (these are equivalent)
            if not ((cube1_four and cube2_sixnine) or (cube2_four and cube1_sixnine)):
                continue

            # Check '81' case
            cube1_eight = (cube1 & eight_mask) == eight_mask
            cube2_eight = (cube2 & eight_mask) == eight_mask

            if not ((cube1_eight and cube2_one) or (cube2_eight and cube1_one)):
                continue

            counter += 1

    # print(n, counter)
    # print(valid_cube_combos)
    return counter


def main():
    sol_81 = ProjectEuler_PathSum_TwoWays_81()
    print(f"sol_81 = {sol_81}")

    sol_82 = ProjectEuler_PathSum_ThreeWays_82()
    print(f"sol_82 = {sol_82}")

    sol_83 = ProjectEuler_PathSum_FourWays_83()
    print(f"sol_83 = {sol_83}")

    sol_84 = ProjectEuler_MonopolyOdds_84()
    print(f"sol_84 = {sol_84}")

    sol_85 = ProjectEuler_CountingRectangles_85()
    print(f"sol_85 = {sol_85}")

    sol_86 = ProjectEuler_CuboidRoute_86()
    print(f"sol_86 = {sol_86}")

    sol_87 = ProjectEuler_PrimePowerTriples_87()
    print(f"sol_87 = {sol_87}")

    sol_88 = ProjectEuler_ProductSum_Numbers_88()
    print(f"sol_88 = {sol_88}")

    sol_89 = ProjectEuler_RomanNumerals_89()
    print(f"sol_89 = {sol_89}")

    sol_90 = ProjectEuler_CubeDigitPairs_90()
    print(f"sol_90 = {sol_90}")


if __name__ == "__main__":
    main()
