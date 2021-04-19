import fenics as pde
import matplotlib.pyplot as plt

from fenics import (
    sin, pi, inner, nabla_grad, grad, div, dx, sym
)
from elasticity_tools import get_true_solution, epsilon, print_error, plot_comparison


def get_solution_with_pressure(mesh, mu, lambda_, f, true_solution):

    P2 = pde.VectorElement("CG", mesh.ufl_cell(), 2)
    P1 = pde.FiniteElement("CG", mesh.ufl_cell(), 1)
    TH = pde.MixedElement([P2, P1])

    V = pde.FunctionSpace(mesh, TH)

    u, p = pde.TrialFunctions(V)
    v, q = pde.TestFunctions(V)

    a = (
        2*mu*inner(epsilon(u), epsilon(v))*dx + p*div(v)*dx 
        - q * div(u) * dx                     + (1/lambda_) * p*q*dx
    )
    L = -inner(f, v) * dx
    bcs = pde.DirichletBC(V.sub(0), pde.project(true_solution, V.sub(0).collapse()), "on_boundary")

    u_lu = pde.Function(V)
    A, b = pde.assemble_system(a, L, bcs=bcs)

    # Solve with LU
    solver = pde.LUSolver()
    solver.set_operator(A)
    solver.solve(u_lu.vector(), b)

    return u_lu.split()[0]

def get_solution_without_pressure(mesh, mu, lambda_, f, true_solution):

    P2 = pde.VectorElement("CG", mesh.ufl_cell(), 2)

    V = pde.FunctionSpace(mesh, P2)

    u = pde.TrialFunction(V)
    v = pde.TestFunction(V)

    a = (
        2*mu*inner(epsilon(u), epsilon(v))*dx + lambda_*div(u)*div(v)*dx
    )
    L = -inner(f, v) * dx
    bcs = pde.DirichletBC(V, true_solution, "on_boundary")

    u_lu = pde.Function(V)
    A, b = pde.assemble_system(a, L, bcs=bcs)

    # Solve with LU
    solver = pde.LUSolver()
    solver.set_operator(A)
    solver.solve(u_lu.vector(), b)

    return u_lu

N = 8
print(f"h = 1/{N}: ")
mesh = pde.UnitSquareMesh(N, N)

solution_space = pde.VectorFunctionSpace(mesh, "CG", 4)
true_solution = get_true_solution(mesh)
    
mu = pde.Constant(1)
lambda_ = pde.Constant(1E3)
f = 2*mu*div(epsilon(true_solution)) + lambda_*grad(div(true_solution))

u_without_pressure = get_solution_without_pressure(mesh, mu, lambda_, f, true_solution)
u_with_pressure = get_solution_with_pressure(mesh, mu, lambda_, f, true_solution)

# Paraview plots:
# pde.File("u_without.pvd") << u_without_pressure
# pde.File("u_with.pvd") << u_with_pressure

print("Without pressure: ")
print_error(u_without_pressure, true_solution, solution_space)
print("With pressure: ")
print_error(u_with_pressure, true_solution, solution_space)

# Comparison plot
plot_comparison(u_without_pressure, u_with_pressure, true_solution)
plt.savefig(f"comparison_h_1_over_{N}.png", dpi=300)
