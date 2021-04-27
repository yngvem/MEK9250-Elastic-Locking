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
    stream_function = 0.01*sin(3*x[0]*x[1]*pi)
    soln = pde.as_vector((
          stream_function.dx(1),
        - stream_function.dx(0)
    ))
    return soln

def calculate_error(u_, soln, soln_space, norm):
    return pde.errornorm(u_, pde.project(soln, soln_space), norm)

def plot_spatial_comparison(u_without_pressure, u_with_pressure, true_solution):

    plt.figure(figsize=(15, 9))

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



def plot_error(lambda_values, N_values, p2_order, errors):

    fig, axes = plt.subplots(2, 2, sharey="row", sharex=True)

    latex_norm = {"L2" : r"$L_2$", "H1" : r"$H_1$"}

    for (i, norm) in enumerate(["L2", "H1"]):
        for (j, label) in enumerate(["without pressure", "with pressure"]):
            for N in N_values:
                err = errors[N][label][norm]
                axes[i][j].plot(lambda_values, err, label=f"h: 1/{N}")
            
            axes[i][j].set_yscale("log")
            axes[i][j].set_xscale("log")

            axes[0][j].set_title(label.capitalize())
            axes[1][j].set_xlabel(r"$\lambda$")
        axes[i][0].set_ylabel(latex_norm[norm])
    
    axes[0][1].legend(loc=1, bbox_to_anchor=[1.6, 1.0])
    plt.tight_layout()
    plt.savefig(f"errors_norms_schemes_order_{p2_order}.pdf", dpi=300)
    plt.show()
