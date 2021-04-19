import fenics as pde
from fenics import sin, pi, sym, grad

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import matplotlib.cm

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

def calculate_error(u_, soln, soln_space, norm):
    return pde.errornorm(u_, pde.project(soln, soln_space), norm)

def plot_comparison(u_without_pressure, u_with_pressure, true_solution):

    plt.figure(figsize=(15, 9)) #, constrained_layout=True)

    norm = LogNorm(vmin=1E-8, vmax=1E-2)
    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='black')

    plt.subplot(231)
    pde.plot(true_solution)
    plt.title("True solution")
    
    plt.subplot(232)
    plt.title("Solution, without pressure")
    pde.plot(u_without_pressure)
    
    plt.subplot(233)
    plt.title("Error, without pressure")
   
    c = pde.plot((u_without_pressure - true_solution)**2, mode="color", norm=norm)
    divider = make_axes_locatable(plt.gca())
    plt.colorbar(c, cax=divider.append_axes("right", size="5%", pad=0.15))

    plt.subplot(235)
    plt.title("Solution, with pressure")
    pde.plot(u_with_pressure)

    plt.subplot(236)
    plt.title("Error, with pressure")
    c = pde.plot((u_with_pressure - true_solution)**2, mode="color", norm=norm)

    divider = make_axes_locatable(plt.gca())
    plt.colorbar(c, cax=divider.append_axes("right", size="5%", pad=0.15))
