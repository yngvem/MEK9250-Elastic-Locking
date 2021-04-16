import fenics as pde
from fenics import sin, pi, sym, grad

def epsilon(x):
    return sym(grad(x))

def get_true_solution(mesh):
    x = pde.SpatialCoordinate(mesh)
    stream_function = sin(x[0]*x[1]*pi)
    soln = pde.as_vector((
          stream_function.dx(1),
        - stream_function.dx(0)
    ))
    return soln

def print_error(u_, soln, soln_space):
    print("  L2", pde.errornorm(u_, pde.project(soln, soln_space), "L2"))
    print("  H1", pde.errornorm(u_, pde.project(soln, soln_space), "H1"))

def plot_comparison(u_lu, u_amg, true_solution):
    plt.figure()
    plt.subplot(231)
    pde.plot(true_solution)
    plt.title("True solution")
    plt.subplot(232)
    plt.title("LU solution")
    pde.plot(u_lu)
    plt.subplot(233)
    plt.title("LU error")
    pde.plot(u_lu - true_solution)
    plt.subplot(234)
    pde.plot(true_solution)
    plt.title("True solution")
    plt.subplot(235)
    plt.title("AMG solution")
    pde.plot(u_amg)
    plt.subplot(236)
    plt.title("AMG error")
    pde.plot(u_amg - true_solution)
