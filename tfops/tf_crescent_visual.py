import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
import numpy as np

customtf = tf.load_op_library('../build/tf_integrated/libCustomFuncsCpu.so')

@ops.RegisterGradient("CrescentSmooth")
def _crescent_grad(op, grad):
  dydx1, dydx2 = customtf.crescent_smooth_gradient(grad, op.inputs[0], op.inputs[1])
  return [dydx1, dydx2]  # List of one Tensor, since we have one input

@ops.RegisterGradient("Crescent")
def _crescent_grad(op, grad):
  dydx1, dydx2 = customtf.crescent_gradient(grad, op.inputs[0], op.inputs[1])
  return [dydx1, dydx2]  # List of one Tensor, since we have one input

@ops.RegisterGradient("SpiralSmooth")
def _crescent_grad(op, grad):
  dydx1, dydx2 = customtf.spiral_smooth_gradient(grad, op.inputs[0], op.inputs[1])
  return [dydx1, dydx2]  # List of one Tensor, since we have one input


@ops.RegisterGradient("Spiral")
def _crescent_grad(op, grad):
  dydx1, dydx2 = customtf.spiral_gradient(grad, op.inputs[0], op.inputs[1])
  return [dydx1, dydx2]  # List of one Tensor, since we have one input

@ops.RegisterGradient("Cb2Smooth")
def _crescent_grad(op, grad):
  dydx1, dydx2 = customtf.cb2_smooth_gradient(grad, op.inputs[0], op.inputs[1])
  return [dydx1, dydx2]  # List of one Tensor, since we have one input


@ops.RegisterGradient("Cb2")
def _crescent_grad(op, grad):
  dydx1, dydx2 = customtf.cb2_gradient(grad, op.inputs[0], op.inputs[1])
  return [dydx1, dydx2]  # List of one Tensor, since we have one input

# crescent
x1 = -1.41831
x2 = 2.0
plrange = [-1.5,2]

# # spiral
# x1 = 1.41831
# x2 = -4.79462
# plrange = [-5,5]

# # CB2
# x1 = 2. #1.0
# x2 = 2.#-0.1
# plrange = [-1,2]

# numit = 1000
# numit = 10000



def get_optimization_path(x1_start, x2_start, numit, smoothable_func, smfactor, opti=tf.keras.optimizers.Adam(learning_rate=1e-2)):
    customtf.set_global_smoothing_factor(smfactor)
    var1, var2 = tf.Variable(x1_start), tf.Variable(x2_start)
    def loss():
       return smoothable_func(var1, var2)
    
    currpath = [[var1.numpy(), var2.numpy()]]
    for i in range(0, numit):            
        opti.minimize(loss=loss, var_list=[var1, var2], tape=tf.GradientTape())     # smoothly descend 
        #if(i%(numit/100) == 0): 
        currpath.append([var1.numpy(), var2.numpy()])

    
    return np.stack(currpath)


numit = 350

# path_discrete = get_optimization_path(x1, x2, numit, crescent_py, np.inf)
path_discrete = get_optimization_path(x1, x2, numit, customtf.crescent, np.inf)

smfactor = 100.
path_smooth = get_optimization_path(x1, x2, numit, customtf.crescent_smooth, smfactor)
#path_smooth = get_optimization_path_adaptive(x1, x2, numit, customtf.crescent_smooth, 5.0)
customtf.set_global_smoothing_factor(smfactor)


# log these values!
print("discrete: ", customtf.crescent(*path_discrete[-1]))
print("smooth: ",customtf.crescent(*path_smooth[-1]))

smfactor_show = 2.

func = customtf.crescent_smooth
def func_grad(x1, x2):
   return customtf.crescent_smooth_gradient(1., x1, x2)
# ===================================================================================================================
# ===================================================================================================================
# ===================================================================================================================
# ===================================================================================================================
# ===================================================================================================================


from utils import run_RnR1_on_mesh, run_RnR2_on_mesh, meshtype
import matplotlib.pyplot as plt

x1range = plrange
x2range = plrange
res_per_dim = 60
mesh = meshtype(x1range, x2range, res_per_dim)

customtf.set_global_smoothing_factor(smfactor_show)
YS_smooth = run_RnR1_on_mesh(func, mesh)
YS_smooth_dx1, YS_smooth_dx2 = run_RnR2_on_mesh(func_grad, mesh)


customtf.set_global_smoothing_factor(np.inf)
YS_discrete = run_RnR1_on_mesh(func, mesh)
YS_discrete_dx1, YS_discrete_dx2 = run_RnR2_on_mesh(func_grad, mesh)

fig = plt.figure(figsize=(9.5,8.5))
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222, projection='3d')
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

X1S, X2S = mesh.as_doublegrid()
ax1.plot_wireframe(X1S, X2S, YS_discrete, alpha=0.5)
ax2.plot_wireframe(X1S, X2S, YS_smooth, alpha=0.5)

ax3.quiver(X1S, X2S, YS_discrete_dx1, YS_discrete_dx2)
ax3.plot(path_discrete[:,0], path_discrete[:,1])

ax4.quiver(X1S, X2S, YS_smooth_dx1, YS_smooth_dx2)
ax4.plot(path_smooth[:,0], path_smooth[:,1])

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xticks(x1range)
    ax.set_yticks(x2range)
    if(ax in [ax1, ax2]): ax.set_zticks([])
    ax.set_xlabel(r'$x_2$')
    ax.set_ylabel(r'$x_1$')

ax4.set_ylabel('')
ax4.set_yticklabels([])
ax3.plot([0.],[0.], 'x',color='red', markersize=7)
ax4.plot([0.],[0.], 'x',color='red', markersize=7)

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.977, top=1., wspace=0.024, hspace = 0.2)
plt.show()
# plt.savefig("figures/smooth_grad_400_marked.svg")  