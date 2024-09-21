# Keshav Pandey
from collections import defaultdict
import decimal
from itertools import combinations, permutations, product
import json
import math
from pathlib import Path

def get_sum_of_square_digits(val: int) -> int:
    result = 0

    while (val):
        result += (val % 10) ** 2
        val //= 10

    return result


def get_sudoku_info(board: list[list[str]], n: int) -> tuple[dict, dict, dict]:
    rows = defaultdict(set)
    cols = defaultdict(set)
    boxes = defaultdict(set)

    # Read in Sudoku puzzle information
    for i in range(n):
        for j in range(n):
            val = board[i][j]

            if (val != "0"):
                rows[i].add(val)
                cols[j].add(val)

                box_idx = (i-(i%3)) + (j // 3)
                boxes[box_idx].add(val)

    return (rows, cols, boxes)


def sudoku_search(board: list[list[str]], n: int, rows: dict, cols: dict, boxes: dict) -> None:
    for i in range(n):
        for j in range(n):
            if board[i][j] != "0":
                continue

            possible_val = None
            box = (i - (i % 3)) + (j // 3)

            for test in range(1, n+1):
                test = str(test)
                if (test not in rows[i]) and (test not in cols[j]) and (test not in boxes[box]):
                    if (possible_val == None):
                        possible_val = test
                    else:
                        break
            else:
                board[i][j] = possible_val
                rows[i].add(possible_val)
                cols[j].add(possible_val)
                boxes[box].add(possible_val)


def sudoku_solver(board: list[list[str]], n: int, rows: dict, cols: dict, boxes: dict, idx: int) -> bool:
    # Find next empty cell that needs to be filled in
    # If no more empty cells, found valid solution and can exit recursion
    if (idx == 81):
        return True

    next_i = idx // n
    next_j = idx % n

    if (board[next_i][next_j] != "0"):
        return sudoku_solver(board, n, rows, cols, boxes, idx+1)

    # Calculate which box current empty cell is in
    box_idx = (next_i-(next_i%3)) + (next_j // 3)

    # Try all possible values for current empty cell
    for test in range(1, n+1):
        test = str(test)

        if (test not in rows[next_i]) and (test not in cols[next_j]) and (test not in boxes[box_idx]):
            board[next_i][next_j] = test
            rows[next_i].add(test)
            cols[next_j].add(test)
            boxes[box_idx].add(test)

            if (sudoku_solver(board, n, rows, cols, boxes, idx+1)):
                return True

            board[next_i][next_j] = "0"
            rows[next_i].remove(test)
            cols[next_j].remove(test)
            boxes[box_idx].remove(test)

    return False


def check_anagramic_squares(sqrs: list, w1: str, w2: str) -> int:
    n = len(sqrs)
    largest_square = -1

    # Get required mapping criteria:
    num_distinct_letters = len(set(w1))

    w1_dict = defaultdict(list)
    for i in range(len(w1)):
        w1_dict[w1[i]].append(i)

    w2_dict = defaultdict(list)
    for i in range(len(w2)):
        w2_dict[w2[i]].append(i)

    # Mapping from w1 to w2 (w1:w2 k:v pairs)
    mapping = dict()

    for key in w1_dict.keys():
        w1_val = tuple(w1_dict[key])
        w2_val = tuple(w2_dict[key])
        mapping[w1_val] = w2_val

    # Check for pair of squares that match this criteria:
    for i in range(n):
        sqr1 = str(sqrs[i])
        if (len(set(sqr1)) != num_distinct_letters):
            continue

        for j in range(i+1, n):
            sqr2 = str(sqrs[j])
            if (len(set(sqr2)) != num_distinct_letters):
                continue

            valid_pair1 = True
            valid_pair2 = True

            for w1_idxs, w2_idxs in mapping.items():
                for w1_idx in w1_idxs:
                    for w2_idx in w2_idxs:
                        if sqr1[w1_idx] != sqr2[w2_idx]:
                            valid_pair1 = False
                            break

            if not valid_pair1:
                for w1_idxs, w2_idxs in mapping.items():
                    for w1_idx in w1_idxs:
                        for w2_idx in w2_idxs:
                            if sqr1[w2_idx] != sqr2[w1_idx]:
                                valid_pair2 = False
                                break

            if valid_pair1 or valid_pair2:
                largest_square = max(largest_square, sqrs[i], sqrs[j])

    return largest_square


def quadratic_solver(n: int):
    discriminant = (2*(n**2)) - (2*n) + 1
    decimal.getcontext().prec = 40
    sqrt_d = decimal.Decimal(discriminant).sqrt()

    if (int(sqrt_d) == sqrt_d):
        # print(f"d={discriminant}, sqrt_d={sqrt_d}")
        return sqrt_d

    return -1


def ProjectEuler_IntegerCoordinateRightTriangles_91() -> int:
    min_coord = 0
    max_coord = 50

    coords = [i for i in range(min_coord, max_coord+1)]
    coord_combos = product(coords, repeat=2)
    triangle_combos = combinations(coord_combos, 2)

    num_triangles = 0
    x0, y0 = 0, 0

    # Iterate over all combinations of points P, Q
    for P, Q in triangle_combos:
        x1, y1 = P
        x2, y2 = Q

        if ((x1, y1) == (x0,y0)):
            continue

        # Calculate vectors for triangle OPQ
        OP_x, OP_y = x1-x0, y1-y0
        OQ_x, OQ_y = x2-x0, y2-y0
        PQ_x, PQ_y = x2-x1, y2-y1

        # Calculate dot product to test if any of the three angles are right angles
        if ((OP_x*OQ_x) + (OP_y*OQ_y)) == 0:
            num_triangles += 1
        elif ((OP_x*PQ_x) + (OP_y*PQ_y)) == 0:
            num_triangles += 1
        elif ((OQ_x*PQ_x) + (OQ_y*PQ_y)) == 0:
            num_triangles += 1

    return num_triangles


# TODO: can be optimized
def ProjectEuler_SquareDigitChains_92() -> int:
    n = 10000000
    num_89ers = 0

    memo_size = get_sum_of_square_digits(n-1)
    memo = [False] * (memo_size+1)

    for i in range(1, memo_size+1):
        curr_val = i
        while (curr_val != 89) and (curr_val != 1):
            curr_val = get_sum_of_square_digits(curr_val)

        if (curr_val == 89):
            memo[i] = True

    for i in range(1, n):
        if memo[get_sum_of_square_digits(i)]:
            num_89ers += 1

    return num_89ers


def ProjectEuler_ArithmeticExpressions_93() -> int:
    def f(a: int, b: int, op_idx: int):
        if (a == -math.inf) or (b == -math.inf): return -math.inf

        if (op_idx == 'a'):
            return a+b
        elif (op_idx == 's'):
            return a-b
        elif (op_idx == 'm'):
            return a*b
        elif (op_idx == 'd'):
            if (b == 0): return -math.inf
            return a/b

    operations = ['a','s','m','d']
    op_combos = list(product(operations, repeat=3))

    best_n = -1
    best_combo = -1

    for a in range(1, 10):
        for b in range(a+1, 10):
            for c in range(b+1, 10):
                for d in range(c+1, 10):
                    digits = [a, b, c, d]
                    digit_combos = list(permutations(digits))
                    obtainable_nums = set()

                    for di in digit_combos:
                        for ops in op_combos:
                            # ab, abc, abcd
                            val1 = f(f(f(di[0], di[1], ops[0]) , di[2], ops[1]), di[3], ops[2])
                            if (val1 < 0): val1 = -math.inf

                            # ab, cd, abcd
                            val2 = f(f(di[0], di[1], ops[0]), f(di[2], di[3], ops[2]), ops[1])
                            if (val2 < 0): val2 = -math.inf

                            # bc, abc, abcd
                            val3 = f(f(di[0], f(di[1], di[2], ops[1]), ops[0]), di[3], ops[2])
                            if (val3 < 0): val3 = -math.inf

                            # bc, bcd, abcd
                            val4 = f(di[0], f(f(di[1], di[2], ops[1]), di[3], ops[2]), ops[1])
                            if (val4 < 0): val4 = -math.inf

                            # cd, bcd, abcd
                            val5 = f(di[0], f(di[1], f(di[2], di[3], ops[2]), ops[1]), ops[0])
                            if (val5 < 0): val5 = -math.inf

                            if (val1 != -math.inf) and (val1 == int(val1)): obtainable_nums.add(val1)
                            if (val2 != -math.inf) and (val2 == int(val2)): obtainable_nums.add(val2)
                            if (val3 != -math.inf) and (val3 == int(val3)): obtainable_nums.add(val3)
                            if (val4 != -math.inf) and (val4 == int(val4)): obtainable_nums.add(val4)
                            if (val5 != -math.inf) and (val5 == int(val5)): obtainable_nums.add(val5)

                    curr_best_n = 0
                    for i in range(1, 100):
                        if (i not in obtainable_nums):
                            curr_best_n = i-1
                            break

                    if (curr_best_n > best_n):
                        best_n = curr_best_n
                        best_combo = (a,b,c,d)

    # Get result string
    result = ""
    for num in best_combo:
        result += str(num)

    return result


def ProjectEuler_AlmostEquilateralTriangles_94() -> int:
    # Initialize constants
    max_L = 1000000000  # 101 317 295
    max_perimeter1_m_val = math.ceil(math.sqrt(max_L / 4))

    # Sub n=1 and solve: (m^2 + 2m - 499999999 = 0)
    # Find: m = 22359.7, -22361.7
    m_limit = 22360
    result = 0

    # Generate all primitive triangles with perimeters <= max_L
    for m in range(1, m_limit):
        start = (m % 2) + 1

        for n in range(start, m, 2):
            if (math.gcd(m, n) != 1):
                continue

            # m_2 = int(math.pow(m, 2))
            # n_2 = int(math.pow(n, 2))
            m_2 = m ** 2
            n_2 = n ** 2

            a = m_2 - n_2
            b = 2 * m * n
            c = m_2 + n_2

            valid_p1 = True if (m <= max_perimeter1_m_val) else False

            # Consider equilateral triangle with 2 c's and 1 double length a
            # if (m <= max_perimeter1_m_val):
            if (valid_p1):
                perimeter1 = 4 * m_2
                init_diff = abs(c - 2*a)

                if (init_diff == 1) and (perimeter1 <= max_L):
                    # print(f"{c,c,a} ; {perimeter1}")
                    result += perimeter1
                    # k = 1 / init_diff
                    # if (int(k) == k):
                    #     print(f"{c,c,a} ; {k}")
                    #     num_p_triangles += 1

            # Consider equilateral triangle with 2 c's and 1 double length b
            perimeter2 = 2 * (c + b)
            if (not valid_p1) and (perimeter2 > max_L): break
            init_diff = abs(c - 2*b)

            if (init_diff == 1) and (perimeter2 <= max_L):
                # print(f"{c,c,b} ; {perimeter2}")
                result += perimeter2
                # k = 1 / init_diff
                # if (int(k) == k):
                #     print(f"{c,c,b} ; {k}")
                #     num_p_triangles += 1


            # if (m <= max_p2_m_val):
            # if (p1 > max_L):
            #     break

            # print(f"{a,b,c} ; {p1}")

            # if (p1 < max_L):
            #     print(f"{a,b,c} ; {k} ; {p1}")
            #     num_p_triangles += 1

            # if (p2 < max_L):
            #     print(f"{a,b,c} ; {k} ; {p2}")
            #     num_p_triangles += 1

    return result


# TODO: can optimize memoization more
def ProjectEuler_AmicableChains_95() -> int:
    # Find longest amicable chain with all values <= n
    n = 1000000
    result = -1
    max_chain_len = -1

    # Initialize memoization table for flagging bad values:
    # memo[i]=True -> bad value ; memo[i]=False -> unchecked value
    # Bad vals = all vals that do not create a chain, or vals that have a proper_divisor_sum > n
    memo = [False] * (n+1)
    memo[1] = True

    # Calculate next element in chain for all numbers up to 'n' using a sieve
    proper_divisor_sum = [1] * (n+1)
    proper_divisor_sum[1] = 0

    for i in range(2, n+1):
        if (proper_divisor_sum[i] > n):
            proper_divisor_sum[i] = 1
            memo[i] = True

        for curr_idx in range(i*2, n+1, i):
            proper_divisor_sum[curr_idx] += i

    # Check every value up to 'n' for an amicable chain
    for i in range(2, n+1):
        if (memo[i]):
            continue

        # A prime number cannot create an amicable chain
        if (proper_divisor_sum[i] == 1):
            memo[i] = True
            continue

        # Try to create an amicable chain starting from 'i'
        curr_chain = set()
        curr_val = i

        while (not memo[curr_val]) and (curr_val not in curr_chain):
            curr_chain.add(curr_val)
            curr_val = proper_divisor_sum[curr_val]

        # If cannot make an amicable chain, flag every value in curr_chain as a bad value
        if (memo[curr_val]):
            for num in curr_chain:
                memo[num] = True

        # If chain is not looping from the very first element, then it is an invalid chain
        # However, the later looping elements in this chain are not bad values
        if (curr_val != i):
            continue

        # Found valid amicable chain, update result variables if necessary
        chain_len = len(curr_chain)

        if (chain_len > max_chain_len):
            result = min(curr_chain)
            max_chain_len = chain_len

    return result


def ProjectEuler_SuDoku_96() -> int:
    try:
        input_file = open(Path("input_files/0096_sudoku.txt"), "r")
        sudokus = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find input Sudoku puzzles")
        return -1

    result = 0

    for curr_board in sudokus:
        # Initialize variables
        n = len(curr_board)
        rows, cols, boxes = get_sudoku_info(curr_board, n)

        # Fill-in numbers with only one possibility:
        sudoku_search(curr_board, n, rows, cols, boxes)

        # Solve Sudoku puzzle using recursive backtracking
        sudoku_solver(curr_board, n, rows, cols, boxes, 0)

        # Increment resultant sum
        top_left_num = (int(curr_board[0][0]) * 100) + (int(curr_board[0][1]) * 10) + int(curr_board[0][2])
        result += top_left_num

    return result


# TODO: can be optimized
def ProjectEuler_Large_NonMersenne_Prime_97() -> int:
    power = 7830457
    const_mult = 28433
    modulo = 10000000000
    result = 1

    # Calculate last 10 digits of math.pow(2, 7830457)
    for _ in range(power):
        result = (result*2) % modulo

    # non_mersenne_prime = (28433 * math.pow(2, 7830457)) + 1
    result = (const_mult * result + 1) % modulo
    return result


def ProjectEuler_AnagramicSquares_98() -> int:
    try:
        input_file = open(Path("input_files/0098_words.txt"), "r")
        words = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find input words")
        return -1

    # Prune all words that do not have an anagram
    my_dict = defaultdict(list)

    for word in words:
        key = ''.join(sorted(word))
        my_dict[key].append(word)

    valid_word_pairs = list()
    for v in my_dict.values():
        if (len(v) > 1):
            valid_word_pairs.append(tuple(v))

    # valid_word_pairs.sort(key = lambda x: len(x[0]), reverse=True)
    longest_word_pair = max(valid_word_pairs, key = lambda x: len(x[0]))
    longest_word_len = len(longest_word_pair[0])

    sorted_word_pairs = [[] for _ in range(longest_word_len+1)]

    for pair in valid_word_pairs:
        pair_len = len(pair[0])
        sorted_word_pairs[pair_len].append(pair)

    largest_square = -1

    for num_digits in range(longest_word_len, -1, -1):
        if len(sorted_word_pairs[num_digits]) == 0:
            continue

        min_sqr = math.ceil(math.sqrt(math.pow(10, num_digits-1)))
        max_sqr = math.ceil(math.sqrt(math.pow(10, num_digits) - 1))

        curr_squares = list()
        for x in range(min_sqr, max_sqr):
            curr_squares.append(int(x ** 2))

        for words in sorted_word_pairs[num_digits]:
            for i in range(len(words)):
                for j in range(i+1, len(words)):
                    w1, w2 = words[i], words[j]
                    result = check_anagramic_squares(curr_squares, w1, w2)
                    largest_square = max(largest_square, result)

        if (largest_square != -1):
            return largest_square

    return largest_square


def ProjectEuler_LargestExponential_99() -> int:
    try:
        input_file = open(Path("input_files/0099_base_exp.txt"), "r")
        exponent_data = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find input exponents")
        return -1

    largest_value = -1
    largest_value_idx = -1
    n = len(exponent_data)

    for i in range(n):
        base, pow = exponent_data[i]
        curr_val = pow * math.log(base)

        if (curr_val > largest_value):
            largest_value = curr_val
            largest_value_idx = i

    return (largest_value_idx+1)


def ProjectEuler_ArrangedProbability_100() -> int:
    # Initialize variables
    curr_b = 15
    curr_n = 21

    gcd_num = 5
    gcd_denom = 7

    max_n = 1000000000000

    # Starting iteration:
    test_b = curr_b + gcd_num
    test_n = -1
    valid_soln = False

    while (test_n < max_n) or (not valid_soln):
        temp = int(test_b / gcd_num)
        test_n = (temp * gcd_denom) + 1

        num_b_discs = quadratic_solver(test_n)
        if (num_b_discs != -1):
            # B = (num_b_discs+1) / 2
            # print(f"test_n={test_n}, test_b={test_b}, B={B}")

            curr_b = test_b
            curr_gcd = math.gcd(test_n, test_b)
            gcd_num = test_b // curr_gcd
            gcd_denom = test_n // curr_gcd
            test_b = curr_b + gcd_num
            valid_soln = True
        else:
            test_b += gcd_num
            valid_soln = False

    # print(type(min_test_n), type(max_test_n), type(gcd_num))
    # print(min_test_n, max_test_n, gcd_num)
    # valid_soln = False


    # for test_n in range(min_test_n, max_test_n, gcd_num):
    #     num_b_discs = quadratic_solver(test_n)
    #     if (num_b_discs != -1):
    #         B = (num_b_discs+1) / 2
    #         print(f"n={test_n}, B={B}")

    # while ():


    # min_n = 21
    # max_n = 100000
    # # min_n = int(math.pow(10, 12))
    # # max_n = int(min_n * 1.5)

    # for n in range(min_n, max_n):
    #     num_b_discs = quadratic_solver(n)
    #     if (num_b_discs != -1):
    #         B = (num_b_discs+1) / 2
    #         print(f"n={n}, B={B}")
    #         # return 5

    return curr_b


def main():
    sol_91 = ProjectEuler_IntegerCoordinateRightTriangles_91()
    print(f"sol_91 = {sol_91}")

    sol_92 = ProjectEuler_SquareDigitChains_92()
    print(f"sol_92 = {sol_92}")

    sol_93 = ProjectEuler_ArithmeticExpressions_93()
    print(f"sol_93 = {sol_93}")

    sol_94 = ProjectEuler_AlmostEquilateralTriangles_94()
    print(f"sol_94 = {sol_94}")

    sol_95 = ProjectEuler_AmicableChains_95()
    print(f"sol_95 = {sol_95}")

    sol_96 = ProjectEuler_SuDoku_96()
    print(f"sol_96 = {sol_96}")

    sol_97 = ProjectEuler_Large_NonMersenne_Prime_97()
    print(f"sol_97 = {sol_97}")

    sol_98 = ProjectEuler_AnagramicSquares_98()
    print(f"sol_98 = {sol_98}")

    sol_99 = ProjectEuler_LargestExponential_99()
    print(f"sol_99 = {sol_99}")

    sol_100 = ProjectEuler_ArrangedProbability_100()
    print(f"sol_100 = {sol_100}")


if __name__ == "__main__":
    main()
