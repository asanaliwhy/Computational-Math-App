import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import bisect, fsolve
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

st.title("Computational Mathematics")

# Sidebar
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

# Function to plot math functions
def plot_function(f, x_range, root=None):
    x = np.linspace(*x_range, 400)
    y = f(x)
    fig, ax = plt.subplots()
    ax.plot(x, y, label="Function")
    ax.axhline(0, color='black', linewidth=1)
    if root:
        ax.scatter(root, f(root), color="red", label="Approximate Root")
    ax.grid(True)
    ax.legend()
    return fig

# Task 1: Graphical Method and Absolute Error
if task == "Graphical Method and Absolute Error":
    def f(x):
        return x**3 - 4*x + 1

    root_numeric = fsolve(f, 2)[0]
    st.write(f" Numerical root: {root_numeric:.6f}")

    approx_root = st.number_input("Enter approximate root:", value=1.8, format="%.4f")

    if st.button("Compute"):
        abs_error = abs(root_numeric - approx_root)
        st.write(f"Absolute Error: {abs_error:.6f}")

        # Update graph
        st.pyplot(plot_function(f, (0, 3), root=approx_root))

# Task 2: Root-Finding Method Comparison
elif task == "Comparison of Root-Finding Methods":
    def f(x):
        return x**2 - 5

    a, b = st.slider("Select interval for bisection:", 0.0, 5.0, (2.0, 3.0))

    if st.button("Compute"):
        root_bisection = bisect(f, a, b)
        st.write(f" Bisection Root: {root_bisection:.6f}")

        def secant_method(f, x0, x1, tol):
            iters = 0
            while abs(x1 - x0) > tol:
                x_temp = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
                x0, x1 = x1, x_temp
                iters += 1
            return x1, iters

        root_secant, secant_iters = secant_method(f, 2, 3, 1e-6)
        st.write(f" Secant Method Root: {root_secant:.6f} (Iterations: {secant_iters})")

        st.pyplot(plot_function(f, (0, 3)))

# Task 3: Jacobi Method
elif task == "Jacobi Method":
    def jacobi(A, b, x0, tol=1e-6, max_iter=100):
        n = len(A)
        x = np.array(x0)
        for _ in range(max_iter):
            x_new = np.copy(x)
            for i in range(n):
                sum_ = sum(A[i][j] * x[j] for j in range(n) if i != j)
                x_new[i] = (b[i] - sum_) / A[i][i]
            if np.linalg.norm(x_new - x, ord=np.inf) < tol:
                return x_new
            x = x_new
        return x

    st.write("Enter matrix A:")
    A = st.data_editor(pd.DataFrame([[1, 1, 1], [0, 2, 5], [2, 3, 1]]), use_container_width=True).to_numpy()
    b = st.data_editor(pd.DataFrame([[6], [-4], [27]]), use_container_width=True).to_numpy().flatten()
    x0 = st.data_editor(pd.DataFrame([[0], [0], [0]]), use_container_width=True).to_numpy().flatten()

    if st.button("Compute"):
        solution = jacobi(A, b, x0)
        st.write(f" Solution: {solution}")

# Task 4: Iterative Matrix Inversion
elif task == "Iterative Method for Matrix Inversion":
    def iterative_inverse(A, tol=1e-6, max_iter=100):
        n = A.shape[0]
        X = np.eye(n) / np.trace(A)
        I = np.eye(n)
        for _ in range(max_iter):
            X_new = X @ (2*I - A @ X)
            if np.linalg.norm(X_new - X, ord=np.inf) < tol:
                return X_new
            X = X_new
        return X

    st.write("Enter matrix A:")
    A = st.data_editor(pd.DataFrame([[4, -2, 1], [-2, 4, -2], [1, -2, 4]]), use_container_width=True).to_numpy()

    if st.button("Compute"):
        A_inv = iterative_inverse(A)
        st.write("🔹 Inverse Matrix:")
        st.write(A_inv)

# Task 5: Linear Curve Fitting
elif task == "Linear Curve Fitting":
    def least_squares(x, y):
        A = np.vstack([x, np.ones(len(x))]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return a, b

    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 5, 7, 11])

    if st.button("Compute"):
        a, b = least_squares(x, y)
        st.write(f" Equation: y = {a:.4f}x + {b:.4f}")

# Task 6: Newton’s Forward Interpolation
elif task == "Newton’s Forward Interpolation":
    x = np.array([0, 1, 2, 3])
    y = np.array([1, 4, 9, 16])

    if st.button("Compute"):
        f_interp = interp1d(x, y, kind="quadratic")
        f_1_5 = f_interp(1.5)
        st.write(f" f(1.5) ≈ {f_1_5:.6f}")

# Task 7: First Derivative (Newton’s Forward Difference)
elif task == "First Derivative Using Newton’s Forward Difference":
    x = np.array([0, 1, 2])
    y = np.array([1, 8, 27])

    if st.button("Compute"):
        dy_dx = (y[2] - y[1]) / (x[2] - x[1])
        st.write(f" dy/dx at x=1 ≈ {dy_dx:.6f}")

# Task 8: Trapezoidal Rule
elif task == "Trapezoidal Rule":
    x = np.linspace(0, 1, 5)
    y = x**2 + x

    if st.button("Compute"):
        integral = trapezoid(y, x)
        st.write(f" Integral: {integral:.6f}")
