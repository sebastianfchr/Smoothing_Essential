import tensorflow as tf
customtf = tf.load_op_library('../build/tf_integrated/libCustomFuncsCpu.so')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import meshtype
import numpy as np
from mpl_toolkits import mplot3d
from utils import run_RnR1_on_mesh, run_RnR2_on_mesh
from tensorflow.python.framework import ops

# import test_tf_register_gradients # where gradients are registered!


customtf = tf.load_op_library('../build/tf_integrated/libCustomFuncsCpu.so')


@ops.RegisterGradient("CrescentSmooth")
def _crescent_grad(op, grad):
  dydx1, dydx1 = customtf.crescent_smooth_gradient(grad, op.inputs[0], op.inputs[1])
  return [dydx1, dydx1]  # List of one Tensor, since we have one input

@ops.RegisterGradient("SpiralSmooth")
def _crescent_grad(op, grad):
  dydx1, dydx1 = customtf.spiral_smooth_gradient(grad, op.inputs[0], op.inputs[1])
  return [dydx1, dydx1]  # List of one Tensor, since we have one input


fig = plt.figure()
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222, projection='3d')
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

# r = [-5,3]
r = [-1.5,2]
x1range = r
x2range = r
# x2range = [0, 2.1]

res_per_dim = 40
mesh = meshtype(x1range, x2range, res_per_dim)

X1S, X2S = mesh.as_doublegrid()

# ============================== run functions on mesh, R^2 -> R and R^2 -> R^2 =========================================

@tf.function
def func(x1,x2):
    return customtf.crescent_smooth(x1,x2)

@tf.function
def func_grad(x1,x2):
    return tf.gradients(func(x1,x2), [x1, x2])

customtf.set_global_smoothing_factor(0.5)

YS_smooth = run_RnR1_on_mesh(func, mesh)
YS_smooth_dx1, YS_smooth_dx2 = run_RnR2_on_mesh(func_grad, mesh)


customtf.set_global_smoothing_factor(np.inf)

YS_discrete = run_RnR1_on_mesh(func, mesh)
YS_discrete_dx1, YS_discrete_dx2 = run_RnR2_on_mesh(func_grad, mesh)


ax1.plot_wireframe(X1S, X2S, YS_discrete, alpha=0.5)
ax2.plot_wireframe(X1S, X2S, YS_smooth, alpha=0.5)


ax4.quiver(X1S, X2S, YS_smooth_dx1, YS_smooth_dx2)
ax3.quiver(X1S, X2S, YS_discrete_dx1, YS_discrete_dx2)


for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xticks(x1range)
    ax.set_yticks(x2range)
    ax.set_xlabel(r'$x_2$')
    ax.set_ylabel(r'$x_1$')

plt.show()
#plt.savefig('example_output.jpg', dpi=250)
