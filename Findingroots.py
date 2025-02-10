import sympy as sp

# Get a polynomial function from user input
def get_polynomial():
    polynomial_input = input("Enter the polynomial (e.g., x**3 - 6*x**2 + 11*x - 6): ")
    x = sp.symbols('x')
    polynomial = sp.sympify(polynomial_input)
    return polynomial, x

# Using remainder theorem
def remainder_theorem(p, x_val, x):
    return p.subs(x, x_val).evalf()

# Newton's Method with root convergence tracking
def newton_method(p, dp, x0, x, tolerate=1e-6, max_iteration=10):
    root_steps = [x0]  # Store intermediate approximations
    for iterate in range(max_iteration):
        f_x = p.subs(x, x0).evalf()
        if abs(f_x) < tolerate:
            return root_steps  # Return the list of convergence steps
        df_x = dp.subs(x, x0).evalf()
        if df_x == 0:
            break  # Avoid division by 0
        x0 = x0 - f_x / df_x
        root_steps.append(x0)  # Append the updated approximation
    return root_steps

# Finding roots using both methods
def find_roots(p, dp, guesses, x, tolerate=1e-6):
    all_roots = {}  # Store root convergence history for each initial guess
    final_roots = []
    for guess in guesses:
        remainder = remainder_theorem(p, guess, x)
        if abs(remainder) < tolerate:
            final_roots.append(round(guess, 6))
            all_roots[guess] = [guess]  # Direct root found, no iterations needed
        else:
            root_steps = newton_method(p, dp, guess, x, tolerate)
            rounded_final_root = round(root_steps[-1], 6)
            if rounded_final_root not in final_roots:
                final_roots.append(rounded_final_root)
            all_roots[guess] = root_steps  # Store iteration steps
    return all_roots, final_roots

if __name__ == "__main__":
    # Getting polynomial via user input
    polynomial, x = get_polynomial()
    derivative = sp.diff(polynomial, x)

    # Initial guesses for roots
    initial_guesses = [float(n) for n in input("Enter initial guesses separated by spaces: ").split()]

    # Find Roots
    root_convergence, final_roots = find_roots(polynomial, derivative, initial_guesses, x)

    # Print root convergence
    for guess, steps in root_convergence.items():
        print(f"Initial guess: {guess}")
        for i, step in enumerate(steps):
            print(f"  Iteration {i}: {step}")
        print(f"  Final root: {steps[-1]}\n")

    print("Final unique root:", final_roots)
