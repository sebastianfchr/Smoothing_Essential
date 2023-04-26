import matplotlib.pyplot as plt
from utils import meshtype
import numpy as np
from utils import run_RnR1_on_mesh, run_RnR2_on_mesh
from build.pybind_integrated.Smoothing import set_smfactor, simple_2d_curve, simple_2d_curve_smooth, simple_2d_curve_grad, simple_2d_curve_smooth_grad, cb2, cb2_smooth, cb2_grad, cb2_smooth_grad, ftest, ftest_grad, ftest_smooth, ftest_smooth_grad


fig = plt.figure()
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222, projection='3d')
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
# ax5 = fig.add_subplot(236)

x1range = [0, 2.1]
x2range = [0, 2.1]

res_per_dim = 40
mesh = meshtype(x1range, x2range, res_per_dim)

# simple_2d_curve_smooth_grad(1., 12.)
# exit(0)
X1S, X2S = mesh.as_doublegrid()

# ============================== run functions on mesh, R^2 -> R and R^2 -> R^2 =========================================


set_smfactor(0.5)  # for infinite sharpness: set_smfactor(math.inf)


# =========================================================
# =========================================================

YS_discrete = run_RnR1_on_mesh(cb2, mesh)
YS_discrete_dx1, YS_discrete_dx2 = run_RnR2_on_mesh(cb2_grad, mesh)


YS_smooth = run_RnR1_on_mesh(cb2_smooth, mesh)
YS_smooth_dx1, YS_smooth_dx2 = run_RnR2_on_mesh(cb2_smooth_grad, mesh)


# YS_dx1_smooth, YS_dx2_smooth = run_RnR2_on_mesh(
#     cb2_grad_smooth, mesh)

# print(YS_smooth_dx1)

ax2.plot_wireframe(X1S, X2S, YS_smooth, alpha=0.5)
ax1.plot_wireframe(X1S, X2S, YS_discrete, alpha=0.5)


ax3.quiver(X1S, X2S, YS_discrete_dx1, YS_discrete_dx2)
ax4.quiver(X1S, X2S, YS_smooth_dx1, YS_smooth_dx2)


for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xticks(x1range)
    ax.set_yticks(x2range)
    ax.set_xlabel(r'$x_2$')
    ax.set_ylabel(r'$x_1$')


# print(np.sum(np.isnan(YS_smooth_dx1)), np.sum(np.isnan(YS_smooth_dx2)))
plt.show()
