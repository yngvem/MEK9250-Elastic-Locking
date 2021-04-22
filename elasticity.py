import fenics as pde
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from fenics import (
    sin, pi, inner, nabla_grad, grad, div, dx, sym
)
from elasticity_tools import get_true_solution, epsilon, plot_spatial_comparison, plot_error

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

N_values = [8, 16, 32, 64, 128]

errors = defaultdict(lambda: defaultdict(lambda: (defaultdict(list))))

mu = pde.Constant(1)
lambda_ = pde.Constant(0)
lambda_values = [1E1, 1E2, 1E3, 1E4, 1E5, 1E6, 1E7, 1E8]

for N in N_values:
    for lambda_value in lambda_values:
        lambda_.assign(lambda_value)
        mesh = pde.UnitSquareMesh(N, N)

        solution_space = pde.VectorFunctionSpace(mesh, "CG", 4)
        true_solution = get_true_solution(mesh)
        true_solution_proj = pde.project(true_solution, solution_space)
            
        f = 2*mu*div(epsilon(true_solution)) + lambda_*grad(div(true_solution))

        u_without_pressure = get_solution_without_pressure(mesh, mu, lambda_, f, true_solution)
        u_with_pressure = get_solution_with_pressure(mesh, mu, lambda_, f, true_solution)

        # Paraview plots:
        # pde.File("u_without.pvd") << u_without_pressure
        # pde.File("u_with.pvd") << u_with_pressure
     
        for (label, u_) in zip(["without pressure", "with pressure"], [u_without_pressure, u_with_pressure]):
            for norm in ["L2", "H1"]:
                error_norm = pde.errornorm(u_, true_solution_proj, norm)
                fun_norm = pde.norm(true_solution_proj, norm)
                errors[N][label][norm].append(error_norm/fun_norm)

        # Spatial plots, not sure what we want to present here?
        #plot_spatial_comparison(u_without_pressure, u_with_pressure, true_solution)
        #plt.savefig(f"comparison_h_1_over_{N}_lambda_{lambda_value}.png", dpi=300)

plot_error(lambda_values, N_values, errors)
