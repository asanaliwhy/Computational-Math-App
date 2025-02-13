import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import bisect, fsolve
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

st.title("Computational Mathematics")

# Sidebar for task selection
task = st.sidebar.selectbox("Choose the task", [
    "Graphical Method and Absolute Error",
    "Comparison of Root-Finding Methods",
    "Jacobi Method",
    "Iterative Method for Matrix Inversion",
    "Linear Curve Fitting",
    "Newton’s Forward Interpolation",
    "First Derivative Using Newton’s Forward Difference",
    "Trapezoidal Rule"
])

st.write(f"### {task}")


# Function to plot functions dynamically
def plot_function(f, x_range, root=None, points=None):
    x = np.linspace(*x_range, 400)
    y = f(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, label="Function", color='blue')
    ax.axhline(0, color='black', linewidth=1, linestyle="--")

    if root is not None:
        ax.scatter(root, f(root), color="red", label="Root")

    if points is not None:
        ax.scatter(points[0], points[1], color="green", label="Data Points")

    ax.grid(True)
    ax.legend()
    return fig


# Task 1: Graphical Method and Absolute Error
if task == "Graphical Method and Absolute Error":
    def f(x):
        return x ** 3 - 4 * x + 1


    root_numeric = round(fsolve(f, 2)[0], 6)
    st.write(f" Numerical root: {root_numeric}")

    approx_root = st.number_input("Enter approximate root:", value=1.8, format="%.4f")

    if st.button("Compute"):
        abs_error = round(abs(root_numeric - approx_root), 6)
        st.write(f"Absolute Error: {abs_error}")

        # Update graph
        st.pyplot(plot_function(f, (0, 3), root=approx_root))

# Task 2: Comparison of Root-Finding Methods
elif task == "Comparison of Root-Finding Methods":
    def f(x):
        return x ** 2 - 5

    a, b = st.slider("Select interval for bisection:", 0.0, 5.0, (2.0, 3.0))

    # Bisection Method with Iteration Count
    def bisection_method(f, a, b, tol=1e-6):
        iter_count = 0
        while abs(b - a) > tol:
            iter_count += 1
            c = (a + b) / 2
            if f(c) == 0 or abs(b - a) < tol:
                return round(c, 6), iter_count
            elif f(a) * f(c) < 0:
                b = c
            else:
                a = c
        return round((a + b) / 2, 6), iter_count

    # Secant Method with Iteration Count
    def secant_method(f, x0, x1, tol=1e-6):
        iters = 0
        while abs(x1 - x0) > tol:
            iters += 1
            x_temp = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
            x0, x1 = x1, x_temp
        return round(x1, 6), iters

    if st.button("Compute"):
        root_bisection, bisection_iters = bisection_method(f, a, b)
        st.write(f" Bisection Root: {root_bisection} ( Iterations: {bisection_iters})")

        root_secant, secant_iters = secant_method(f, 2, 3)
        st.write(f" Secant Root: {root_secant} ( Iterations: {secant_iters})")

        # Plot function
        st.pyplot(plot_function(f, (0, 3)))

# Task 3: Jacobi Method
elif task == "Jacobi Method":
    def is_diagonally_dominant(A):
        """Check if a matrix is diagonally dominant."""
        for i in range(len(A)):
            if abs(A[i][i]) < sum(abs(A[i][j]) for j in range(len(A)) if j != i):
                return False
        return True

    def jacobi(A, b, x0, tol=1e-6, max_iter=100):
        """Jacobi method for solving Ax = b"""
        if not is_diagonally_dominant(A):
            return "Error: Matrix is not diagonally dominant. Jacobi method may not converge."

        n = len(A)
        x = np.array(x0, dtype=float)

        for _ in range(max_iter):
            x_new = np.copy(x)
            for i in range(n):
                sum_ = sum(A[i][j] * x[j] for j in range(n) if i != j)
                x_new[i] = (b[i] - sum_) / A[i][i]

            if np.linalg.norm(x_new - x, ord=np.inf) < tol:
                return np.round(x_new, 6)  # Return rounded values

            x = x_new

        return "Error: Maximum iterations reached. Method did not converge."

    # Get user input for matrix A, vector b, and initial guess x0
    st.write("Enter matrix A:")
    A = st.data_editor(pd.DataFrame([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])).to_numpy()

    st.write("Enter vector b:")
    b = st.data_editor(pd.DataFrame([[2], [6], [2]])).to_numpy().flatten()

    st.write("Enter initial guess x0:")
    x0 = st.data_editor(pd.DataFrame([[0], [0], [0]])).to_numpy().flatten()

    if st.button("Compute"):
        solution = jacobi(A, b, x0)
        st.write(f"Solution: {solution}")

# Task 4: Iterative Method for Matrix Inversion
elif task == "Iterative Method for Matrix Inversion":
    def iterative_inverse(A, tol=1e-6, max_iter=100):
        """Iterative method to compute matrix inverse"""
        n = A.shape[0]
        X = np.eye(n) / np.trace(A)  # Initial guess
        I = np.eye(n)

        for _ in range(max_iter):
            X_new = X @ (2 * I - A @ X)
            if np.linalg.norm(X_new - X, ord=np.inf) < tol:
                return np.round(X_new, 2)  # Rounded to 2 decimal places
            X = X_new

        return "Error: Maximum iterations reached. Method did not converge."

    st.write("Enter matrix A:")
    A = st.data_editor(pd.DataFrame([[4, -2, 1], [-2, 4, -2], [1, -2, 4]])).to_numpy()

    if st.button("Compute"):
        A_inv = iterative_inverse(A)

        if isinstance(A_inv, str):
            st.write(A_inv)  # Display error message if no convergence
        else:
            st.write("Inverse Matrix:")
            st.dataframe(pd.DataFrame(A_inv))  # Display properly formatted table


# Task 5: Linear Curve Fitting
elif task == "Linear Curve Fitting":
    st.write("Linear regression for given data points.")

    # Define example dataset
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 5, 7, 11])

    # Show data table
    df = pd.DataFrame({"x": x, "y": y})
    st.write("### Data Points:")
    st.dataframe(df)

    if st.button("Compute"):
        # Perform linear regression
        A = np.vstack([x, np.ones(len(x))]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]  # Compute slope & intercept

        # Display equation
        st.write(f"**Equation:** y = {round(a, 4)}x + {round(b, 4)}")

        # Generate fitted line
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = a * x_fit + b

        # Plot data points and fitted line
        fig, ax = plt.subplots()
        ax.scatter(x, y, color='red', label="Data Points")  # Original points
        ax.plot(x_fit, y_fit, color='blue', label="Fitted Line")  # Linear regression line
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid()

        # Display graph
        st.pyplot(fig)


# Task 6: Newton’s Forward Interpolation
if task == "Newton’s Forward Interpolation":
    st.write("#### Newton’s Forward Interpolation Formula")

    x_values = st.text_area("Enter x values (comma-separated):", "0,1,2,3")
    y_values = st.text_area("Enter y values (comma-separated):", "1,4,9,16")

    x = np.array([float(i) for i in x_values.split(",")])
    y = np.array([float(i) for i in y_values.split(",")])

    def divided_differences(x, y):
        n = len(y)
        coef = np.zeros([n, n])
        coef[:, 0] = y

        for j in range(1, n):
            for i in range(n - j):
                coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])

        return coef[0, :]

    coef = divided_differences(x, y)

    st.write("### Divided Difference Table")
    st.write(pd.DataFrame(coef.reshape(1, -1)))

    st.latex(r"f(x) = f_0 + f_1(x - x_0) + f_2(x - x_0)(x - x_1) + ...")

    x_pred = st.number_input("Enter x value to estimate f(x):", value=1.5)

    def newton_interpolation(x, y, x_pred):
        coef = divided_differences(x, y)
        n = len(x)
        result = coef[0]
        term = 1.0
        for i in range(1, n):
            term *= (x_pred - x[i - 1])
            result += coef[i] * term
        return result

    if st.button("Compute"):
        f_pred = round(newton_interpolation(x, y, x_pred), 6)
        st.write(f"Estimated f({x_pred}) = {f_pred}")

        # Visualization
        f_interp = interp1d(x, y, kind="quadratic")
        x_plot = np.linspace(min(x), max(x), 100)
        y_plot = f_interp(x_plot)

        fig, ax = plt.subplots()
        ax.plot(x_plot, y_plot, label="Interpolated Curve", color='blue')
        ax.scatter(x, y, color="red", label="Data Points")
        ax.scatter(x_pred, f_pred, color="green", label=f"Estimated f({x_pred})")

        ax.legend()
        ax.grid()
        st.pyplot(fig)


# Task 7: First Derivative Using Newton’s Forward Difference
if task == "First Derivative Using Newton’s Forward Difference":
    st.write("#### Newton’s Forward Difference Formula for First Derivative")

    x_values = st.text_area("Enter x values (comma-separated):", "0,1,2")
    y_values = st.text_area("Enter y values (comma-separated):", "1,8,27")

    x = np.array([float(i) for i in x_values.split(",")])
    y = np.array([float(i) for i in y_values.split(",")])

    def forward_differences(y):
        """Compute forward differences table."""
        n = len(y)
        table = np.zeros((n, n))
        table[:, 0] = y

        for j in range(1, n):
            for i in range(n - j):
                table[i, j] = table[i + 1, j - 1] - table[i, j - 1]

        return table

    diff_table = forward_differences(y)

    st.write("### Forward Difference Table")
    st.write(pd.DataFrame(diff_table))

    st.latex(r"\frac{dy}{dx} \approx \frac{f(x_1) - f(x_0)}{x_1 - x_0}")

    x_target = st.number_input("Enter x value to estimate derivative:", value=1.0)

    def newton_forward_derivative(x, y, x_target):
        """Compute first derivative using Newton’s forward difference formula."""
        diff_table = forward_differences(y)
        h = x[1] - x[0]
        return diff_table[0, 1] / h

    if st.button("Compute"):
        derivative = round(newton_forward_derivative(x, y, x_target), 6)
        st.write(f"Estimated dy/dx at x={x_target}: {derivative}")

        # Visualization
        f_interp = interp1d(x, y, kind="quadratic")
        x_plot = np.linspace(min(x), max(x), 100)
        y_plot = f_interp(x_plot)

        fig, ax = plt.subplots()
        ax.plot(x_plot, y_plot, label="Interpolated Curve", color='blue')
        ax.scatter(x, y, color="red", label="Data Points")
        ax.scatter(x_target, newton_forward_derivative(x, y, x_target), color="green", label=f"dy/dx at x={x_target}")

        ax.legend()
        ax.grid()
        st.pyplot(fig)


# Task 8: Trapezoidal Rule
if task == "Trapezoidal Rule":
    st.write("#### Numerical Integration Using the Trapezoidal Rule")

    st.latex(r"\int_{a}^{b} f(x) dx \approx \frac{h}{2} \sum_{i=0}^{n} (f(x_i) + f(x_{i+1}))")

    # Define function
    def f(x):
        return x ** 2 + x

    a = st.number_input("Enter lower limit (a):", value=0.0)
    b = st.number_input("Enter upper limit (b):", value=1.0)
    n = st.number_input("Enter number of subintervals:", value=4, step=1, min_value=1)

    x_exact = np.linspace(a, b, 100)
    y_exact = f(x_exact)

    def trapezoidal_rule(f, a, b, n):
        """Compute integral using Trapezoidal Rule."""
        x = np.linspace(a, b, n + 1)
        y = f(x)
        h = (b - a) / n
        integral = (h / 2) * (y[0] + 2 * sum(y[1:-1]) + y[-1])
        return integral

    if st.button("Compute"):
        approx_integral = round(trapezoidal_rule(f, a, b, n), 6)
        exact_integral = round((b**3 / 3 + b**2 / 2) - (a**3 / 3 + a**2 / 2), 6)

        st.write(f"**Approximate Integral:** {approx_integral}")
        st.write(f"**Exact Integral:** {exact_integral}")
        st.write(f"**Error:** {abs(exact_integral - approx_integral)}")

        # Visualization
        x_trapezoids = np.linspace(a, b, n + 1)
        y_trapezoids = f(x_trapezoids)

        fig, ax = plt.subplots()
        ax.plot(x_exact, y_exact, label="Function", color="blue")
        ax.fill_between(x_trapezoids, y_trapezoids, color='gray', alpha=0.4, label="Trapezoidal Approximation")
        ax.scatter(x_trapezoids, y_trapezoids, color="red", label="Subinterval Points")

        ax.legend()
        ax.grid()
        st.pyplot(fig)


