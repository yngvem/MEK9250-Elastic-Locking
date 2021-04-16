import fenics as pde
import matplotlib.pyplot as plt

from fenics import (
    sin, pi, inner, nabla_grad, grad, div, dx, sym
)
from elasticity_tools import get_true_solution, epsilon, print_error, plot_comparison


mesh = pde.UnitSquareMesh(32, 32)
P2 = pde.VectorElement("CG", mesh.ufl_cell(), 2)
P1 = pde.FiniteElement("CG", mesh.ufl_cell(), 1)
TH = pde.MixedElement([P2, P1])

V = pde.FunctionSpace(mesh, TH)
solution_space = pde.VectorFunctionSpace(mesh, "CG", 4)
soln = get_true_solution(mesh)

mu = pde.Constant(1)
lambda_ = pde.Constant(1e3)
f = 2*mu*div(epsilon(soln)) + lambda_*grad(div(soln))

u, p = pde.TrialFunctions(V)
v, q = pde.TestFunctions(V)

a = (
    2*mu*inner(epsilon(u), epsilon(v))*dx + p*div(v)*dx 
    - q * div(u) * dx                     + (1/lambda_) * p*q*dx
)
L = -inner(f, v) * dx
bcs = pde.DirichletBC(V.sub(0), pde.project(soln, solution_space), "on_boundary")

u_lu = pde.Function(V)
u_amg = pde.Function(V)
A, b = pde.assemble_system(a, L, bcs=bcs)

# Solve with LU
solver = pde.LUSolver()
solver.set_operator(A)
solver.solve(u_lu.vector(), b)
print_error(u_lu.split()[0], soln, solution_space)

# Solve with AMG
#solver = pde.KrylovSolver('minres', 'amg')
#solver.set_operator(A)
#solver.solve(u_amg.vector(), b)
#print_error(u_amg.split()[0], soln, solution_space)

# Show comparison
plot_comparison(u_lu, u_amg, true_solution)
plt.show()
