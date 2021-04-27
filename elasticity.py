import fenics as pde
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from fenics import (
    sin, pi, inner, nabla_grad, grad, div, dx, sym
)
from elasticity_tools import get_true_solution, epsilon, plot_spatial_comparison, plot_error

def get_solution_with_pressure(mesh, mu, lambda_, f, true_solution, p2_order):

    P2 = pde.VectorElement("CG", mesh.ufl_cell(), p2_order)
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

    u_lu = pde.Function(V, name="Displacement, with pressure")
    A, b = pde.assemble_system(a, L, bcs=bcs)

    # Solve with LU
    solver = pde.LUSolver()
    solver.set_operator(A)
    solver.solve(u_lu.vector(), b)

    return u_lu.split()[0]

def get_solution_without_pressure(mesh, mu, lambda_, f, true_solution, p2_order):

    P2 = pde.VectorElement("CG", mesh.ufl_cell(), p2_order)

    V = pde.FunctionSpace(mesh, P2)

    u = pde.TrialFunction(V)
    v = pde.TestFunction(V)

    a = (
        2*mu*inner(epsilon(u), epsilon(v))*dx + lambda_*div(u)*div(v)*dx
    )
    L = -inner(f, v) * dx
    bcs = pde.DirichletBC(V, true_solution, "on_boundary")

    u_lu = pde.Function(V, name="Displacement, without pressure")
    A, b = pde.assemble_system(a, L, bcs=bcs)

    # Solve with LU
    solver = pde.LUSolver()
    solver.set_operator(A)
    solver.solve(u_lu.vector(), b)

    return u_lu

N_values = [8, 16, 32, 64, 128]

mu = pde.Constant(1)
lambda_ = pde.Constant(0)
lambda_values = [1E1, 1E2, 1E3, 1E4, 1E5, 1E6, 1E7, 1E8, 1E9, 1E10, 1E11, 1E12]

for p2_order in [1, 2]:
    errors = defaultdict(lambda: defaultdict(lambda: (defaultdict(list))))
    
    for N in N_values:
        for lambda_value in lambda_values:
            lambda_.assign(lambda_value)
            mesh = pde.UnitSquareMesh(N, N)

            solution_space = pde.VectorFunctionSpace(mesh, "CG", 4)
            true_solution = get_true_solution(mesh)
            true_solution_proj = pde.project(true_solution, solution_space)

            f = 2*mu*div(epsilon(true_solution)) + lambda_*grad(div(true_solution))

            u_without_pressure = get_solution_without_pressure(mesh, mu, lambda_, f, true_solution, p2_order)
            u_with_pressure = get_solution_with_pressure(mesh, mu, lambda_, f, true_solution, p2_order)

            for (label, u_) in zip(["without pressure", "with pressure"], [u_without_pressure, u_with_pressure]):
                for norm in ["L2", "H1"]:
                    error_norm = pde.errornorm(u_, true_solution_proj, norm)
                    fun_norm = pde.norm(true_solution_proj, norm)
                    errors[N][label][norm].append(error_norm/fun_norm)


    plot_error(lambda_values, N_values, p2_order, errors)


lambda_values = np.linspace(1, 10**12, 200)

for N in [8, 128]:
    print("N: ", N)
    mesh = pde.UnitSquareMesh(N, N)
    error_space = pde.FunctionSpace(mesh, "CG", 2)
    error_without = pde.Function(error_space, name="Error, withput pressure")
    error_with = pde.Function(error_space, name="Error, with pressure")
    
    pvd_u_without = pde.File(f"solutions_lin/resolution_{N}/u_without_pressure.pvd")
    pvd_u_with = pde.File(f"solutions_lin/resolution_{N}/u_with_pressure.pvd")
    pvd_error_without = pde.File(f"solutions_lin/resolution_{N}/error_without_pressure.pvd")
    pvd_error_with = pde.File(f"solutions_lin/resolution_{N}/error_with_pressure.pvd")
     
    solution_space = pde.VectorFunctionSpace(mesh, "CG", 4)
    true_solution = get_true_solution(mesh)
    true_solution_proj = pde.project(true_solution, solution_space)
    pde.File(f"solutions_lin/resolution_{N}/u_analytic.pvd") << true_solution_proj

    for (i, lambda_value) in enumerate(lambda_values):
        if i%10==0:
            print(f"Step {i}/{len(lambda_values)}")
        lambda_.assign(lambda_value)
    
        f = 2*mu*div(epsilon(true_solution)) + lambda_*grad(div(true_solution))

        u_without_pressure = get_solution_without_pressure(mesh, mu, lambda_, f, true_solution)
        u_with_pressure = get_solution_with_pressure(mesh, mu, lambda_, f, true_solution)

        pvd_u_without << u_without_pressure
        pvd_u_with << u_with_pressure
            
        pde.assign(error_without, pde.project(pde.sqrt((u_without_pressure - true_solution)**2), error_space))
        pde.assign(error_with, pde.project(pde.sqrt((u_with_pressure - true_solution)**2), error_space))

        pvd_error_without << error_without
        pvd_error_with << error_with
