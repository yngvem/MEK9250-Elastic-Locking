import fenics as pde
import matplotlib.pyplot as plt

from fenics import (
    sin, pi, inner, nabla_grad, grad, div, dx, sym
)
from elasticity_tools import get_true_solution, epsilon, print_error, plot_comparison


mesh = pde.UnitSquareMesh(32, 32)
P2 = pde.VectorElement("CG", mesh.ufl_cell(), 2)

V = pde.FunctionSpace(mesh, P2)
soln_space = pde.VectorFunctionSpace(mesh, "CG", 4)
soln = get_true_solution(mesh)

mu = pde.Constant(1)
lambda_ = pde.Constant(1e3)
f = 2*mu*div(epsilon(soln)) + lambda_*grad(div(soln))

u = pde.TrialFunction(V)
v = pde.TestFunction(V)

a = (
    2*mu*inner(epsilon(u), epsilon(v))*dx + lambda_*div(u)*div(v)*dx
)
L = -inner(f, v) * dx
bcs = pde.DirichletBC(V, soln, "on_boundary")

u_lu = pde.Function(V)
u_amg = pde.Function(V)
A, b = pde.assemble_system(a, L, bcs=bcs)

# Solve with LU
solver = pde.LUSolver()
solver.set_operator(A)
solver.solve(u_lu.vector(), b)
print_error(u_lu, soln, soln_space)


# Solve with AMG
solver = pde.KrylovSolver('cg', 'amg')
solver.set_operator(A)
solver.solve(u_amg.vector(), b)
print("AMG-L2", pde.errornorm(u_amg, pde.project(soln, soln_space), "L2"))
print("AMG-H1", pde.errornorm(u_amg, pde.project(soln, soln_space), "H1"), flush=True)

plot_comparison(u_lu, u_amg, true_solution)
plt.show()
