# Keshav Pandey
from collections import defaultdict
from itertools import combinations_with_replacement, permutations, product
import json
import math
from pathlib import Path

# Get sum of all digits in 'val' squared. Ex: 145 = 1^2 + 4^2 + 5^2 = 42
def get_sum_of_square_digits(val: int) -> int:
    result = 0

    while (val):
        digit = val % 10
        result += digit * digit
        val //= 10

    return result


# Concatenate list of digits into an integer. Ex: [1,3,4] = 134
def concat_digits(digits: list) -> int:
    result = 0
    for digit in digits:
        result = (result * 10) + digit
    return result


def sqr_digit_chains(combos: list, r: int, memo: list, factorials: list) -> int:
    num_89_chains = 0

    # Iterate over all r-length combinations of digits
    for combo in combos:
        # Convert list of digits into an integer value
        curr_val = concat_digits(combo)

        # If chain for curr_val arrives at '89', calculate num permutations of curr digit 'combo'
        if memo[get_sum_of_square_digits(curr_val)]:
            permutations = factorials[r]

            # Account for repeated digits
            digit_count = [0] * 10
            for digit in combo:
                digit_count[digit] += 1

            for count in digit_count:
                if (count > 1):
                    permutations //= factorials[count]

            # Account for invalid permutations with a leading '0'
            if (digit_count[0] > 0):
                permutations = (permutations * (r-digit_count[0])) // r

            # Increment result by number of unique permutations
            num_89_chains += permutations

    return num_89_chains


def get_sudoku_info(board: list[list[int]], n: int) -> tuple[list, list, list]:
    rows = [[False] * (n+1) for _ in range(n)]
    cols = [[False] * (n+1) for _ in range(n)]
    boxes = [[False] * (n+1) for _ in range(n)]

    # Read in Sudoku puzzle information
    for i in range(n):
        for j in range(n):
            val = board[i][j]

            if (val != 0):
                rows[i][val] = True
                cols[j][val] = True

                box_idx = (i - (i % 3)) + (j // 3)
                boxes[box_idx][val] = True

    return (rows, cols, boxes)


def sudoku_search(board: list[list[int]], n: int, rows: list, cols: list, boxes: list) -> None:
    check_sudoku = True

    # Each time we find at least one number that updates the board, we reset check_sudoku=True
    # (Because filling in one sqr might mean a previously checked sqr now also has a soln)
    while (check_sudoku):
        check_sudoku = False

        # Iterate over all squares in Sudoku puzzle
        for i in range(n):
            for j in range(n):
                if board[i][j] != 0:
                    continue

                box_idx = (i - (i % 3)) + (j // 3)
                only_soln = None

                # Check nums 1-9 and see if only one of these values is valid for curr square
                for num in range(1, n+1):
                    if not (rows[i][num] or cols[j][num] or boxes[box_idx][num]):
                        if (only_soln == None):
                            only_soln = num
                        else:
                            break
                else:
                    board[i][j] = only_soln
                    rows[i][only_soln] = True
                    cols[j][only_soln] = True
                    boxes[box_idx][only_soln] = True
                    check_sudoku = True


def sudoku_solver(board: list[list[int]], n: int, rows: list, cols: list, boxes: list, idx: int) -> bool:
    # Find next empty cell that needs to be filled in
    # If no more empty cells, found valid solution and can exit recursion
    if (idx == 81):
        return True

    i = idx // n
    j = idx % n

    # If current cell is already filled in, skip to next cell in puzzle
    if (board[i][j] != 0):
        return sudoku_solver(board, n, rows, cols, boxes, idx+1)

    # Calculate which box current empty cell is in
    box_idx = (i - (i % 3)) + (j // 3)

    # Try all possible values for current empty cell
    for num in range(1, n+1):
        if not (rows[i][num] or cols[j][num] or boxes[box_idx][num]):
            board[i][j] = num
            rows[i][num] = True
            cols[j][num] = True
            boxes[box_idx][num] = True

            if (sudoku_solver(board, n, rows, cols, boxes, idx+1)):
                return True

            board[i][j] = 0
            rows[i][num] = False
            cols[j][num] = False
            boxes[box_idx][num] = False

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


def ProjectEuler_IntegerCoordinateRightTriangles_91() -> int:
    # Initialize variables to define problem
    min_coord = 0
    max_coord = 50

    # Store count of number of triangles possible with right angle at point P
    num_P_triangles = 0

    # Iterate over all combinations of points P, Q where P=(x1,y1) and Q=(x2,y2)
    # Note: we set loops such that point P is always above and to the left of point Q
    for x1 in range(min_coord, max_coord+1):
        for x2 in range(x1, max_coord+1):
            for y1 in range(min_coord, max_coord+1):
                for y2 in range(min_coord, y1+1):
                    if ((x1, y1) == (0, 0)):
                        break

                    if ((x1, y1) == (x2, y2)):
                        continue

                    # Calculate vectors of triangle OPQ that go through point P
                    OP_x, OP_y = x1, y1
                    PQ_x, PQ_y = x2-x1, y2-y1

                    # Calculate dot product to test if point P is a right angle
                    if ((OP_x*PQ_x) + (OP_y*PQ_y)) == 0:
                        num_P_triangles += 1

    # Total triangles with right angle at point O will be (max_coord^2)
    # Total triangles with right angle at point Q will be equal to total triangles at point P
    num_O_triangles = int(math.pow(max_coord, 2))
    num_triangles = num_O_triangles + (2 * num_P_triangles)

    return num_triangles


def ProjectEuler_SquareDigitChains_92() -> int:
    # Initialize variables to define problem
    n = 10000000
    num_89ers = 0

    # Store result for all square digit sums given (num < n) in memoization table:
    memo_size = get_sum_of_square_digits(n-1)
    memo = [False] * (memo_size+1)

    for i in range(1, memo_size+1):
        curr_val = i
        while (curr_val != 89) and (curr_val != 1):
            curr_val = get_sum_of_square_digits(curr_val)

        if (curr_val == 89):
            memo[i] = True

    # Note: sum_of_square_digits(974322) = sum_of_square_digits(223479) = 163
    # This is because the result depends on the digits in each num, and not the order of the digits
    # So rather than checking all num < n, we check all combinations of digits <= max_digits
    digits = [9,8,7,6,5,4,3,2,1,0]
    max_digits = 7

    # Pre-calculate factorial value for necessary digits
    factorials = [1] * (max_digits+1)
    for i in range(2, max_digits+1):
        factorials[i] = i * factorials[i-1]

    # Iterate over all combinations of numbers with only 1 digit up to 7 digits
    # (We pop last combination as '0' is invalid combination with respect to problem description)
    for i in range(1, max_digits+1):
        curr_combos = list(combinations_with_replacement(digits, i))
        curr_combos.pop()
        num_89ers += sqr_digit_chains(curr_combos, i, memo, factorials)

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
    # Find all "almost equilateral" triangles with perimeter <= max_p
    max_p = 1000000000
    result = 0

    # Store side lengths of first primitive Pythagorean triple, and the perimeter of the
    # "almost equilateral" triangle that it forms
    a = 3
    b = 4
    c = 5
    perimeter = 2*(c+a)

    # Alternate between Pythagorean triple transformations to directly find the next solution
    # This pattern occurs due to the nature of the Pell Equation that describes this problem:
    # x^2 - 3b^2 = 1, where x = 3a +/- 2
    transform1 = False

    # Generate larger and larger primitive triangle solutions until max_p is reached
    while (perimeter <= max_p):
        result += perimeter

        # Generate next triangle using either transformation 1 or transformation 3
        if transform1:
            a, b, c = (a)-(2*b)+(2*c), (2*a)-(b)+(2*c), (2*a)-(2*b)+(3*c)
            transform1 = False
        else:
            a, b, c = (-a)+(2*b)+(2*c), (-2*a)+(b)+(2*c), (-2*a)+(2*b)+(3*c)
            transform1 = True

        # The "almost equilateral" will be formed with hypotenuse and minimum side length
        if (a < b):
            perimeter = 2*(c+a)
        else:
            perimeter = 2*(c+b)

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
        # TODO: add elif (memo[proper_divisor_sum[i]] == True): memo[i] = True
        # But check if this adds any noticeably performance improvement first, since
        # Often times sum(proper_divisors[i]) > i anyways
        # TODO: add check for prime numbers here rather than in next loop?
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


# TODO: can optimize further by adding more logic checking
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
        top_left_num = (curr_board[0][0] * 100) + (curr_board[0][1] * 10) + curr_board[0][2]
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
    # Search for first valid disc arrangement with total_discs > n
    n = 1000000000000

    # Initialize variables to values of first valid arrangement of discs
    blue_discs = 15
    total_discs = 21

    # Ratio of blue_discs:total_discs in reduced form = gcd_num:gcd_denom
    gcd_num = 5
    gcd_denom = 7

    two_fold = True

    # While current disc arrangement has too few discs, get next arrangement
    while (total_discs <= n):
        # Solve for next valid number of total_discs and blue_discs
        if (two_fold):
            total_discs = 2 * (gcd_num + gcd_denom) * gcd_num
            blue_discs = ((total_discs - 1) * gcd_num) // gcd_denom
            two_fold = False
        else:
            total_discs = (gcd_num + gcd_denom) * gcd_num
            blue_discs = ((total_discs - 1) * gcd_num) // gcd_denom
            two_fold = True

        # Update ratio variables for next iteration
        curr_gcd = math.gcd(total_discs, blue_discs)
        gcd_num = blue_discs // curr_gcd
        gcd_denom = total_discs // curr_gcd

    return blue_discs


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
