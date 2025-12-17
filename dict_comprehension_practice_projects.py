"""
Dictionary Comprehension - Real-World Mini Projects
====================================================
Practice exercises to master dictionary comprehension!

Instructions:
1. Solve each project using DICTIONARY COMPREHENSION
2. Run the file to check your outputs against expected results
3. Try to solve WITHOUT looking at the tutorial first
4. Hints are provided if you get stuck

Why practice these? They represent REAL scenarios you'll encounter in:
- Web development, Data analysis, APIs, File management, E-commerce, etc.
"""

# ============================================================================
# PROJECT 1: Grade Calculator
# ============================================================================
print("=" * 60)
print("PROJECT 1: Grade Calculator")
print("=" * 60)

# Given: List of students and their scores
students_scores = [
    ('Alice', 85),
    ('Bob', 72),
    ('Charlie', 90),
    ('David', 65),
    ('Eve', 78)
]

# TODO: Create a dictionary that converts scores to letter grades
# Grading scale: A (85+), B (75-84), C (65-74), F (below 65)
# Expected output: {'Alice': 'A', 'Bob': 'C', 'Charlie': 'A', 'David': 'C', 'Eve': 'B'}

# HINT: Use nested if-else (ternary operators) in your comprehension
# HINT: Syntax is - {key: value_if_true if condition else value_if_false for item in iterable}

grades = {}  # Replace this with your dictionary comprehension

print("Your answer:", grades)
print("Expected:   {'Alice': 'A', 'Bob': 'C', 'Charlie': 'A', 'David': 'C', 'Eve': 'B'}")
print()


# ============================================================================
# PROJECT 2: Word Frequency Counter
# ============================================================================
print("=" * 60)
print("PROJECT 2: Word Frequency Counter")
print("=" * 60)

sentence = "the quick brown fox jumps over the lazy dog the fox"

# TODO: Count how many times each word appears in the sentence
# Expected output: {'the': 3, 'quick': 1, 'brown': 1, 'fox': 2, 'jumps': 1, 'over': 1, 'lazy': 1, 'dog': 1}

# HINT: Use sentence.split() to get a list of words
# HINT: Use set(sentence.split()) to get unique words
# HINT: Use sentence.split().count(word) to count occurrences

word_count = {}  # Replace this with your dictionary comprehension

print("Your answer:", word_count)
print("Expected:   The word 'the' appears 3 times, 'fox' appears 2 times")
print()


# ============================================================================
# PROJECT 3: API Response Parser (In-Stock Products Only)
# ============================================================================
print("=" * 60)
print("PROJECT 3: API Response Parser")
print("=" * 60)

# Simulating data from an e-commerce API
api_data = [
    {'id': 101, 'name': 'Product A', 'price': 29.99, 'stock': 50},
    {'id': 102, 'name': 'Product B', 'price': 49.99, 'stock': 0},
    {'id': 103, 'name': 'Product C', 'price': 19.99, 'stock': 120},
    {'id': 104, 'name': 'Product D', 'price': 39.99, 'stock': 0},
]

# TODO: Create a dictionary with product ID as key, and name+price as value
# BUT only include products that are in stock (stock > 0)
# Expected output: {101: {'name': 'Product A', 'price': 29.99}, 103: {'name': 'Product C', 'price': 19.99}}

# HINT: Loop through api_data
# HINT: Add an if condition to filter stock > 0
# HINT: Use item['id'] for key, and a dictionary for value

in_stock_products = {}  # Replace this with your dictionary comprehension

print("Your answer:", in_stock_products)
print("Expected:   Only products 101 and 103 (the ones with stock > 0)")
print()


# ============================================================================
# PROJECT 4: Configuration File Generator
# ============================================================================
print("=" * 60)
print("PROJECT 4: Environment Config Generator")
print("=" * 60)

base_settings = ['database_url', 'api_key', 'debug_mode', 'max_connections']

# TODO: Generate a development configuration where each setting is prefixed with "DEV_" and uppercased
# Expected output: {'database_url': 'DEV_DATABASE_URL', 'api_key': 'DEV_API_KEY', ...}

# HINT: Use f-strings to format: f"DEV_{setting.upper()}"
# HINT: The key is the original setting, the value is the formatted version

dev_config = {}  # Replace this with your dictionary comprehension

print("Your answer:", dev_config)
print("Expected:   Each value should be 'DEV_' + uppercase version of the key")
print()


# ============================================================================
# PROJECT 5: URL Parameter Parser
# ============================================================================
print("=" * 60)
print("PROJECT 5: URL Query Parameter Parser")
print("=" * 60)

url_params = "?name=John&age=30&city=NYC&active=true"

# TODO: Parse the URL query string into a dictionary
# Expected output: {'name': 'John', 'age': '30', 'city': 'NYC', 'active': 'true'}

# HINT: Remove the '?' first: url_params[1:]
# HINT: Split by '&' to get pairs: .split('&')
# HINT: For each pair, split by '=' to get key and value
# HINT: pair.split('=')[0] is key, pair.split('=')[1] is value

params = {}  # Replace this with your dictionary comprehension

print("Your answer:", params)
print("Expected:   {'name': 'John', 'age': '30', 'city': 'NYC', 'active': 'true'}")
print()


# ============================================================================
# PROJECT 6: Temperature Converter (Celsius to Fahrenheit)
# ============================================================================
print("=" * 60)
print("PROJECT 6: Temperature Converter")
print("=" * 60)

temps_celsius = {'Monday': 20, 'Tuesday': 22, 'Wednesday': 19, 'Thursday': 25, 'Friday': 18}

# TODO: Convert all temperatures from Celsius to Fahrenheit
# Formula: F = (C * 9/5) + 32
# Expected output: {'Monday': 68.0, 'Tuesday': 71.6, 'Wednesday': 66.2, 'Thursday': 77.0, 'Friday': 64.4}

# HINT: Use .items() to get (day, temp) pairs
# HINT: Keep the day as key, apply formula to temp for value

temps_fahrenheit = {}  # Replace this with your dictionary comprehension

print("Your answer:", temps_fahrenheit)
print("Expected:   Monday should be 68.0°F, Thursday should be 77.0°F")
print()


# ============================================================================
# PROJECT 7: Filter and Transform - Even Squares Only
# ============================================================================
print("=" * 60)
print("PROJECT 7: Even Number Squares")
print("=" * 60)

numbers = range(1, 11)  # Numbers 1 through 10

# TODO: Create a dictionary with ONLY even numbers as keys and their squares as values
# Expected output: {2: 4, 4: 16, 6: 36, 8: 64, 10: 100}

# HINT: Use an if condition to filter: if number % 2 == 0
# HINT: Key is the number, value is number ** 2

even_squares = {}  # Replace this with your dictionary comprehension

print("Your answer:", even_squares)
print("Expected:   {2: 4, 4: 16, 6: 36, 8: 64, 10: 100}")
print()


# ============================================================================
# BONUS PROJECT 8: Inventory Management System
# ============================================================================
print("=" * 60)
print("BONUS: Inventory Management (Multiple Tasks)")
print("=" * 60)

inventory = [
    ('Laptop', 1200, 5),
    ('Mouse', 25, 50),
    ('Keyboard', 75, 30),
    ('Monitor', 300, 10),
    ('Webcam', 150, 15),
]

# TODO 8A: Create a price lookup dictionary (name -> price)
# Expected: {'Laptop': 1200, 'Mouse': 25, 'Keyboard': 75, 'Monitor': 300, 'Webcam': 150}

price_lookup = {}  # Replace this

print("8A - Price Lookup:")
print("Your answer:", price_lookup)
print()

# TODO 8B: Find items that need restocking (stock < 20)
# Show name -> current stock level
# Expected: {'Laptop': 5, 'Monitor': 10, 'Webcam': 15}

restock_needed = {}  # Replace this

print("8B - Restock Needed:")
print("Your answer:", restock_needed)
print()

# TODO 8C: Calculate total value per item (price * stock)
# Expected: {'Laptop': 6000, 'Mouse': 1250, 'Keyboard': 2250, 'Monitor': 3000, 'Webcam': 2250}

inventory_value = {}  # Replace this

print("8C - Inventory Value:")
print("Your answer:", inventory_value)
print()

# TODO 8D: Apply 10% discount to expensive items (price > 100), keep others same
# Expected: {'Laptop': 1080.0, 'Mouse': 25, 'Keyboard': 75, 'Monitor': 270.0, 'Webcam': 135.0}

discounted_prices = {}  # Replace this

print("8D - Discounted Prices:")
print("Your answer:", discounted_prices)
print()


# ============================================================================
# ADVANCED CHALLENGE: Nested Dictionary Comprehension
# ============================================================================
print("=" * 60)
print("ADVANCED: Student Grade Summary")
print("=" * 60)

students = {
    'Alice': [85, 90, 88],
    'Bob': [70, 75, 72],
    'Charlie': [95, 92, 98]
}

# TODO: Create a nested dictionary with:
# - Student name as key
# - Value is another dictionary with 'average' and 'grade' keys
# Expected output:
# {
#   'Alice': {'average': 87.67, 'grade': 'A'},
#   'Bob': {'average': 72.33, 'grade': 'C'},
#   'Charlie': {'average': 95.0, 'grade': 'A'}
# }

# HINT: Use students.items() to get (name, scores) pairs
# HINT: Calculate average: sum(scores) / len(scores)
# HINT: Grade logic: A (85+), B (75-84), C (65-74), F (below 65)

student_summary = {}  # Replace this with your nested dictionary comprehension

print("Your answer:", student_summary)
print("Expected:   Each student should have 'average' and 'grade' keys")
print()


print("=" * 60)
print("All projects completed! Check your answers above.")
print("=" * 60)
print()
print("Next steps:")
print("1. Compare your solutions with expected outputs")
print("2. If stuck, review dictionary_comprehension_tutorial.py")
print("3. Try modifying the data and re-running your solutions")
print("4. Challenge: Can you solve each in ONE LINE?")
