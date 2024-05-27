def add(num1, num2):
  """Adds two numbers and returns the result."""
  return num1 + num2

def subtract(num1, num2):
  """Subtracts two numbers and returns the result."""
  return num1 - num2

def multiply(num1, num2):
  """Multiplies two numbers and returns the result."""
  return num1 * num2

def divide(num1, num2):
  """Divides two numbers and returns the result. Handles division by zero error."""
  if num2 == 0:
    return "Error: Cannot divide by zero"
  return num1 / num2

while True:
  # Get user input for operation
  print("Enter operation (add, subtract, multiply, divide): ")
  operation = input().lower()

  # Get user input for numbers
  print("Enter first number: ")
  num1 = float(input())
  print("Enter second number: ")
  num2 = float(input())

  # Perform calculation based on operation
  result = None
  if operation == "add":
    result = add(num1, num2)
  elif operation == "subtract":
    result = subtract(num1, num2)
  elif operation == "multiply":
    result = multiply(num1, num2)
  elif operation == "divide":
    result = divide(num1, num2)
  else:
    print("Invalid operation. Please try again.")

  # Display result
  if result is not None:
    print("Result:", result)

  # Ask user if they want to continue
  print("Do you want to perform another calculation? (y/n)")
  choice = input().lower()
  if choice != "y":
    break

print("Thank you for using the calculator!")

