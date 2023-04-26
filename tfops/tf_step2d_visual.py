import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
import numpy as np

customtf = tf.load_op_library('../build/tf_integrated/libCustomFuncsCpu.so')

@ops.RegisterGradient("StepSmooth")
def _step_grad(op, grad):
  dydx1, dydx2 = customtf.step_smooth_gradient(grad, op.inputs[0], op.inputs[1])
  return [dydx1, dydx2]  # List of one Tensor, since we have one input

@ops.RegisterGradient("Step")
def _step_grad(op, grad):
  dydx1, dydx2 = customtf.step_gradient(grad, op.inputs[0], op.inputs[1])
  return [dydx1, dydx2]  # List of one Tensor, since we have one input


# # step
# x1 = 2.
# x2 = -2.
# step
x1 = -1.
x2 = 2.
plrange = [-2,2]


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

# path_discrete = get_optimization_path(x1, x2, numit, step_py, np.inf)
path_discrete = get_optimization_path(x1, x2, numit, customtf.step, np.inf)

smfactor = 3.
path_smooth = get_optimization_path(x1, x2, numit, customtf.step_smooth, smfactor)
#path_smooth = get_optimization_path_adaptive(x1, x2, numit, customtf.step_smooth, 5.0)
customtf.set_global_smoothing_factor(smfactor)


# log these values!
print("discrete: ", customtf.step(*path_discrete[-1]))
print("smooth: ",customtf.step(*path_smooth[-1]))

#exit(0)

smfactor_show_mesh = 5.
smfactor_show_gradient = smfactor

func = customtf.step_smooth
def func_grad(x1, x2):
   return customtf.step_smooth_gradient(1., x1, x2)
# ===================================================================================================================
# ===================================================================================================================
# ===================================================================================================================
# ===================================================================================================================
# ===================================================================================================================



from utils import run_RnR1_on_mesh, run_RnR2_on_mesh, meshtype, run_RnR1_on_mesh_discontinuous
import matplotlib.pyplot as plt

x1range = plrange
x2range = plrange
res_per_dim = 60
mesh = meshtype(x1range, x2range, res_per_dim)

customtf.set_global_smoothing_factor(smfactor_show_mesh)
YS_smooth = run_RnR1_on_mesh(func, mesh)
customtf.set_global_smoothing_factor(smfactor_show_gradient)
YS_smooth_dx1, YS_smooth_dx2 = run_RnR2_on_mesh(func_grad, mesh)


customtf.set_global_smoothing_factor(np.inf)
# in the 3d plot, mask the points matplotlib interpolates the mesh
maskfunc = lambda x1,x2 : x1*x1+x2*x2 < 2 - 0.08 or x1*x1+x2*x2 > 2 + 0.08 
YS_discrete = run_RnR1_on_mesh_discontinuous(func, mesh, maskfunc)
YS_discrete_dx1, YS_discrete_dx2 = run_RnR2_on_mesh(func_grad, mesh)

fig = plt.figure(figsize=(9.5,8.5))
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222, projection='3d')
# ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

X1S, X2S = mesh.as_doublegrid()
ax1.plot_wireframe(X1S, X2S, YS_discrete, alpha=0.4)
# ax1.plot_wireframe(X1S, X2S, YS_discrete, alpha=0.32, color='black')
ax1.view_init(elev=16, azim=-136)
ax2.plot_wireframe(X1S, X2S, YS_smooth, alpha=0.4)
# ax2.plot_wireframe(X1S, X2S, YS_smooth, alpha=0.28, color = 'black')
ax2.view_init(elev=16, azim=-136)

# ax3.quiver(X1S, X2S, YS_discrete_dx1, YS_discrete_dx2, alpha=0.7)
# ax3.plot(path_discrete[:,0], path_discrete[:,1])

ax4.quiver(X1S, X2S, YS_smooth_dx1, YS_smooth_dx2, alpha=0.8)
l = ax4.plot(path_smooth[:,0], path_smooth[:,1], alpha=0.7)
ax4.plot(path_smooth[-1,0], path_smooth[-1,1], 'x', color='red', markersize=7)
# l = ax4.plot(path_smooth[:,0], path_smooth[:,1], '-', color='black', alpha=0.7)
# ax4.plot(path_smooth[-1,0], path_smooth[-1,1], 'x', color='black', markersize=7)
# visutils.add_arrow(l[0], position=-2, color='gray')

for ax in [ax1, ax2, ax4]:
    ax.set_xticks(x1range)
    ax.set_yticks(x2range)
    if(ax in [ax1, ax2]): ax.set_zticks([])
    ax.set_xlabel(r'$x_2$')
    ax.set_ylabel(r'$x_1$')

ax4.set_ylabel('')
ax4.set_yticklabels([])


plt.subplots_adjust(left=0.1, bottom=0.1, right=0.977, top=1., wspace=0.024, hspace = 0.2)
# plt.show()
plt.savefig("/home/seb/Desktop/write/smoothing/content/figures/discontinuous_opt_func_and_grad.svg")  