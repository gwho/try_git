"""
Dictionary Comprehension Tutorial
==================================
A beginner-friendly guide to understanding dictionary comprehension in Python
"""

# ============================================================================
# EXAMPLE 1: Basic Dictionary Comprehension
# ============================================================================

# Traditional way using a for loop
traditional_squares = {}
for number in range(1, 6):
    traditional_squares[number] = number ** 2

print("Traditional approach:", traditional_squares)

# Dictionary comprehension way - does the same thing in one line!
# Syntax: {key_expression: value_expression for item in iterable}
comprehension_squares = {number: number ** 2 for number in range(1, 6)}

print("Comprehension approach:", comprehension_squares)
# Output: {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}


# ============================================================================
# EXAMPLE 2: Creating a Dictionary from Two Lists
# ============================================================================

# We have two lists: one for keys, one for values
fruits = ['apple', 'banana', 'cherry', 'date']
prices = [1.20, 0.50, 2.30, 3.00]

# Using zip() to pair them together in dictionary comprehension
# zip() takes two lists and pairs up their elements
fruit_prices = {fruit: price for fruit, price in zip(fruits, prices)}

print("\nFruit prices:", fruit_prices)
# Output: {'apple': 1.20, 'banana': 0.50, 'cherry': 2.30, 'date': 3.00}


# ============================================================================
# EXAMPLE 3: Dictionary Comprehension with Conditional (if statement)
# ============================================================================

# Let's say we only want even numbers and their squares
# Syntax: {key: value for item in iterable if condition}
even_squares = {number: number ** 2 for number in range(1, 11) if number % 2 == 0}

print("\nEven number squares:", even_squares)
# Output: {2: 4, 4: 16, 6: 36, 8: 64, 10: 100}


# ============================================================================
# EXAMPLE 4: Transforming an Existing Dictionary
# ============================================================================

# Original dictionary with temperatures in Celsius
temps_celsius = {'Monday': 20, 'Tuesday': 22, 'Wednesday': 19, 'Thursday': 25}

# Convert all temperatures to Fahrenheit using formula: F = (C * 9/5) + 32
temps_fahrenheit = {day: (temp * 9/5) + 32 for day, temp in temps_celsius.items()}

print("\nTemperatures in Fahrenheit:", temps_fahrenheit)
# Output: {'Monday': 68.0, 'Tuesday': 71.6, 'Wednesday': 66.2, 'Thursday': 77.0}


# ============================================================================
# EXAMPLE 5: Using String Methods in Dictionary Comprehension
# ============================================================================

# Create a dictionary with words and their lengths
words = ['hello', 'world', 'python', 'code']

# The key is the uppercase word, the value is its length
word_info = {word.upper(): len(word) for word in words}

print("\nWord lengths:", word_info)
# Output: {'HELLO': 5, 'WORLD': 5, 'PYTHON': 6, 'CODE': 4}


# ============================================================================
# EXAMPLE 6: Conditional with if-else (Ternary Operator)
# ============================================================================

# Create a dictionary that categorizes numbers as 'even' or 'odd'
# Syntax: {key: value_if_true if condition else value_if_false for item in iterable}
number_types = {num: 'even' if num % 2 == 0 else 'odd' for num in range(1, 6)}

print("\nNumber classifications:", number_types)
# Output: {1: 'odd', 2: 'even', 3: 'odd', 4: 'even', 5: 'odd'}


# ============================================================================
# EXAMPLE 7: Nested Dictionary Comprehension (Advanced)
# ============================================================================

# Create a multiplication table as a nested dictionary
# Outer dictionary: numbers 1-3, Inner dictionary: that number multiplied by 1-3
multiplication_table = {
    i: {j: i * j for j in range(1, 4)}
    for i in range(1, 4)
}

print("\nMultiplication table:", multiplication_table)
# Output: {1: {1: 1, 2: 2, 3: 3}, 2: {1: 2, 2: 4, 3: 6}, 3: {1: 3, 2: 6, 3: 9}}
