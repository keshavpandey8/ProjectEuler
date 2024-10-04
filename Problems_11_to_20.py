# Keshav Pandey
import json
import math
from pathlib import Path

# Helper function that finds largest product of 'k' adjacent elements in a list
def largest_product_in_list(nums: list, k: int) -> int:
    # Iterate through ith digit one-by-one
    i = 0
    n = len(nums)
    curr_product = 1
    largest_product = -1
    calcNewProduct = True

    while (i < n):
        if (calcNewProduct):
            curr_product = 1
            num_vals = k

            while (num_vals > 0) and (i < n):
                curr_digit = int(nums[i])
                if (curr_digit == 0):
                    curr_product = 1
                    num_vals = k
                    i += 1
                    continue
                else:
                    curr_product *= curr_digit
                    num_vals -= 1
                    i += 1

            if (num_vals == 0):
                largest_product = max(curr_product, largest_product)
            calcNewProduct = False
        else:
            curr_digit = int(nums[i])
            if (curr_digit == 0):
                calcNewProduct = True
                i += 1
            else:
                # Multiply next val and divide off prev. val to get next 'k' digit product
                curr_product = (curr_product / int(nums[i - k])) * curr_digit
                largest_product = max(curr_product, largest_product)
                i += 1
    return int(largest_product)


# Helper function that returns number of divisors for positive input integer 'num'
def get_num_divisors(num: int, primes: list) -> int:
    if (num <= 0):
        return -1

    num_divisors = 1
    limit = math.sqrt(num)
    p_factor_exp = 1
    i = 0

    while (num > 1) and (primes[i] <= limit):
        while ((num % primes[i]) == 0):
            num //= primes[i]
            p_factor_exp += 1
        else:
            num_divisors *= p_factor_exp
            p_factor_exp = 1
            limit = math.sqrt(num)
        i += 1

    if (num > 1):
        num_divisors *= 2

    return num_divisors


def collatz_chain_len(num: int, memo_table: list, len_table: int) -> int:
    # If already found chain_len for 'num', return tabulated solution
    if (num < len_table) and (memo_table[num] != -1):
        return memo_table[num]

    # Continue recursing until end of chain is found
    if (num % 2):
        next_num = (3*num + 1) // 2
        curr_chain_len = collatz_chain_len(next_num, memo_table, len_table) + 2
    else:
        next_num = num // 2
        curr_chain_len = collatz_chain_len(next_num, memo_table, len_table) + 1

    # Insert current solution in memoization table
    if (num < len_table):
        memo_table[num] = curr_chain_len

    return curr_chain_len


class Date:
    days_in_month = {1:31, 2:28, 2.1:29, 3:31, 4:30, 5:31,\
                     6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

    def __init__(self, day: int, month: int, year: int, weekday: int):
        self.day = day
        self.month = month
        self.year = year

        # Define: Mon = 1, Tues = 2, ..., Sun = 7
        self.weekday = weekday
        self.leap_year = self.is_leap_year()

        if (self.leap_year) and (self.month == 2):
            self.month = 2.1

        self.num_day_in_month = self.days_in_month.get(self.month)

    def __lt__(self, other):
        if (self.year < other.year):
            return True
        elif (self.year > other.year):
            return False
        else:
            if (self.month < other.month):
                return True
            elif (self.month > other.month):
                return False
            else:
                return (self.day < other.day)

    def __eq__(self, other):
        return (self.year == other.year) and (self.month == other.month) and (self.day == other.day)

    def __le__(self, other):
        return (self.__lt__(other)) or (self.__eq__(other))

    def is_leap_year(self) -> bool:
        if ((self.year % 4) == 0):
            if ((self.year % 100) != 0) or (((self.year % 100) == 0) and ((self.year % 400) == 0)):
                return True
        return False

    def increment_day(self) -> None:
        # Next day is new year
        if (self.month == 12) and (self.day == self.num_day_in_month):
            # Update day and weekday
            self.day = 1
            if (self.weekday == 7):
                self.weekday = 1
            else:
                self.weekday += 1

            # Update month
            self.month = 1
            self.num_day_in_month = self.days_in_month.get(self.month)

            # Update year
            self.year += 1
            self.leap_year = self.is_leap_year()

        # Next day is new month
        elif (self.day == self.num_day_in_month):
            # Update day and weekday
            self.day = 1
            if (self.weekday == 7):
                self.weekday = 1
            else:
                self.weekday += 1

            # Update month
            self.month = math.floor(self.month + 1)
            if (self.leap_year) and (self.month == 2):
                self.month = 2.1
            self.num_day_in_month = self.days_in_month.get(self.month)

        # Next day is in same month and year
        else:
            # Update day and weekday
            self.day += 1
            if (self.weekday == 7):
                self.weekday = 1
            else:
                self.weekday += 1

    def __str__(self):
        print_info = f"{self.day}//{self.month}//{self.year}"
        return print_info


def ProjectEuler_LargestGridProduct_11() -> int:
    # Read 20x20 grid of integers
    try:
        input_file = open(Path("input_files/0011_grid.txt"), "r")
        input_grid = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find 20x20 grid of integers")
        return -1

    # Initialize variables
    row_len = len(input_grid[0])
    col_len = len(input_grid)
    k = 4
    largest_product = -1

    # Find greatest product of 'k' horizontally adjacent numbers:
    for row in input_grid:
        largest_product = max(largest_product, largest_product_in_list(row, k))

    # Find greatest product of 'k' vertically adjacent numbers:
    for i in range(len(input_grid)):
        col = [0] * len(input_grid[i])
        for j in range(len(input_grid[i])):
            col[j] = input_grid[j][i]
        largest_product = max(largest_product, largest_product_in_list(col, k))

    # Find greatest product of 'k' right diagonally adjacent numbers starting from top row:
    for i in range(row_len):
        curr_diag = list()
        curr_i = i
        curr_j = 0
        while (curr_i < row_len) and (curr_j < col_len):
            curr_diag.append(input_grid[curr_j][curr_i])
            curr_i += 1
            curr_j += 1
        largest_product = max(largest_product, largest_product_in_list(curr_diag, k))

    # Find greatest product of 'k' left diagonally adjacent numbers starting from top row:
    for i in range(row_len):
        curr_diag = list()
        curr_i = i
        curr_j = 0
        while (curr_i >= 0) and (curr_j < col_len):
            curr_diag.append(input_grid[curr_j][curr_i])
            curr_i -= 1
            curr_j += 1
        largest_product = max(largest_product, largest_product_in_list(curr_diag, k))

    # Find greatest product of 'k' right diagonally adjacent numbers starting from bottom row:
    for i in range(row_len):
        curr_diag = list()
        curr_i = i
        curr_j = col_len - 1
        while (curr_i < row_len) and (curr_j >= 0):
            curr_diag.append(input_grid[curr_j][curr_i])
            curr_i += 1
            curr_j -= 1
        largest_product = max(largest_product, largest_product_in_list(curr_diag, k))

    # Find greatest product of 'k' left diagonally adjacent numbers starting from bottom row:
    for i in range(row_len):
        curr_diag = list()
        curr_i = i
        curr_j = col_len - 1
        while (curr_i >= 0) and (curr_j >= 0):
            curr_diag.append(input_grid[curr_j][curr_i])
            curr_i -= 1
            curr_j -= 1
        largest_product = max(largest_product, largest_product_in_list(curr_diag, k))

    return largest_product


def ProjectEuler_DivisibleTriangularNumber_12() -> int:
    try:
        input_file = open(Path("precomputed_primes/primes_10_thousand.txt"), "r")
        primes_list = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find list of prime numbers")
        return -1

    # Want to find first number with at least 'n' divisors
    n = 500

    # Initialize variables
    curr_triangle_num = 1
    curr_natural_num = 1

    # Iterate through triangle numbers from smallest to largest until find solution
    while (get_num_divisors(curr_triangle_num, primes_list) < n):
        curr_natural_num += 1
        curr_triangle_num += curr_natural_num

    return curr_triangle_num


# TODO: optimize / solve mathematically
def ProjectEuler_LargeSum_13() -> int:
    # Read list of one hundred 50-digit integers
    try:
        input_file = open(Path("input_files/0013_nums.txt"), "r")
        input_list = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find list of one hundred integers")
        return -1

    sum = 0
    for val in input_list:
        sum += val

    sum = str(sum)[:10]
    return sum


def ProjectEuler_LongestCollatzSequence_14() -> int:
    # Find longest Collatz sequence starting from a value less than 'n'
    n = 1000000

    # Initialize variables
    longest_chain_len = 0
    best_starting_num = -1

    # Chain length from any given number is always the same
    # So, store prev. calculated results in array (memoization)
    prev_results = [-1] * n
    prev_results[1] = 1

    # An even value '2k' will have Collatz(2k) = Collatz(k) + 1
    # So for all (i < n/2), Collatz(i) cannot be the longest sequences
    for i in range (n//2, n):
        if (collatz_chain_len(i, prev_results, n) > longest_chain_len):
            best_starting_num = i
            longest_chain_len = prev_results[i]

    return best_starting_num


def ProjectEuler_LatticePaths_15() -> int:
    # Find number of routes in 20x20 grid
    n = 20

    # Permutation problem:
    # How many different ways can we order 'n' moves right and 'n' moves down
    num_routes = int((math.factorial(2*n)) / (math.factorial(n) * math.factorial(n)))
    return num_routes


def ProjectEuler_PowerDigitSum_16() -> int:
    # Find sum of all digits in 2^n
    n = 1000

    # Initialize variables
    num = 1 << n
    sum = 0

    # Sum all digits in 'num' from right to left
    while (num > 0):
        right_digit = num % 10
        sum += right_digit
        num = num // 10

    return sum


def ProjectEuler_NumberLetterCounts_17() -> int:
    # If a number is less than 20, it will be one of these unique words
    # Note: smallest expected num is '1' and a zero digit is never said (ex: '20') so 0 = ""
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",\
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]

    # If a number is in the tens, it will have one of these prefixes
    tens = ["", "", "twenty", "thirty","forty","fifty","sixty","seventy","eighty","ninety"]

    # If a number is in the hundreds, must add 10 letters for "hundredand"
    hundred = len("hundredand")

    # Max expected 'n' for this problem is 1000
    one_thousand = len("onethousand")

    # Convert string representations to number of letters
    for i in range(len(ones)):
        ones[i] = len(ones[i])

    for i in range(len(tens)):
        tens[i] = len(tens[i])

    # Want total number of letters needed to write out all numbers from 1 to n (inclusive)
    n = 1000
    total_letters = 0

    # Accumulate num_letters required for each value from 1 to n (inclusive)
    for i in range(1, n+1):
        if (i <= 0) or (i > 1000):
            print(f"Error: negative values and values greater than 1000 not supported")
            return -1
        elif (i <= 19):
            total_letters += ones[i]
        elif (i <= 99):
            total_letters += (tens[(i//10)] + ones[i%10])
        elif (i <= 999):
            total_letters += ones[(i//100)] + hundred
            tens_val = i % 100

            if (tens_val == 0):
                total_letters -= 3  # If value is exactly 100, 200, etc. then don't need word "and"
            if (tens_val <= 19):
                total_letters += ones[tens_val]
            else:
                total_letters += (tens[(tens_val//10)] + ones[tens_val%10])
        elif (i == 1000):
            total_letters += one_thousand

    return total_letters


def ProjectEuler_MaximumPathSumI_18() -> int:
    # Read 15 row triangle:
    try:
        input_file = open(Path("input_files/0018_triangle.txt"), "r")
        input_triangle = json.load(input_file)
        input_file.close()
    except FileNotFoundError:
        print(f"Error: could not find 15 row triangle")
        return -1

    # Create dp matrix to store subproblem solutions
    dp = [([-math.inf] * len(input_triangle[-1])) for _ in range(len(input_triangle))]

    # Initialize dp matrix with base case (one row triangle)
    dp[0][0] = input_triangle[0][0]

    # Iterate through input triangle and fill out dp matrix with subproblem solutions
    for i in range(1, len(input_triangle)):
        for j in range(0, len(input_triangle[i])):
            # First path to current location in triangle is prev. row with same j co-ordinate
            first_path = dp[i-1][j]

            # Second path to current location in triangle is prev. row with prev. j co-ordinate
            if (j == 0):
                second_path = -math.inf
            else:
                second_path = dp[i-1][j-1]

            # Choose best path to current location then increment by current value
            dp[i][j] = max(first_path, second_path) + input_triangle[i][j]

    # Find largest path_sum in last row of triangle, then return
    max_path_sum = max(dp[-1])
    return max_path_sum


def ProjectEuler_CountingSundays_19() -> int:
    # Find total number of Sundays on first of month between start and end date
    start_day = 1
    start_month = 1
    start_year = 1901

    end_day = 31
    end_month = 12
    end_year = 2000

    # Determine which day of week the start_date is
    # Known information:
        # Jan 1st, 1900 = Monday
        # 1900 was not a leap year -> 365 days between 1/1/1900 and 1/1/1901
        # Define: Mon = 1, Tues = 2, ..., Sun = 7
    start_weekday = 1 + (365 % 7)

    # Initialize variables
    today = Date(start_day, start_month, start_year, start_weekday)
    end_date = Date(end_day, end_month, end_year, None)
    num_sundays = 0

    # Iterate "today" up to "end_date" and increment num_sundays as appropriate
    while (today <= end_date):
        if (today.weekday == 7) and (today.day == 1):
            num_sundays += 1
        today.increment_day()

    return num_sundays


def ProjectEuler_FactorialDigitSum_20() -> int:
    # Calculate sum of digits in n!
    n = 100

    # Initialize variables
    num = math.factorial(n)
    sum = 0

    # Sum all digits in 'num' from right to left
    while (num > 0):
        right_digit = num % 10
        sum += right_digit
        num = num // 10

    return sum


def main():
    sol_11 = ProjectEuler_LargestGridProduct_11()
    print(f"sol_11 = {sol_11}")

    # TODO: can optimize futher using co-prime identity
    sol_12 = ProjectEuler_DivisibleTriangularNumber_12()
    print(f"sol_12 = {sol_12}")

    sol_13 = ProjectEuler_LargeSum_13()
    print(f"sol_13 = {sol_13}")

    sol_14 = ProjectEuler_LongestCollatzSequence_14()
    print(f"sol_14 = {sol_14}")

    sol_15 = ProjectEuler_LatticePaths_15()
    print(f"sol_15 = {sol_15}")

    sol_16 = ProjectEuler_PowerDigitSum_16()
    print(f"sol_16 = {sol_16}")

    sol_17 = ProjectEuler_NumberLetterCounts_17()
    print(f"sol_17 = {sol_17}")

    sol_18 = ProjectEuler_MaximumPathSumI_18()
    print(f"sol_18 = {sol_18}")

    sol_19 = ProjectEuler_CountingSundays_19()
    print(f"sol_19 = {sol_19}")

    sol_20 = ProjectEuler_FactorialDigitSum_20()
    print(f"sol_20 = {sol_20}")


if __name__ == "__main__":
    main()
