import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
import numpy as np

customtf = tf.load_op_library('../build/tf_integrated/libCustomFuncsCpu.so')

@ops.RegisterGradient("Cb2Smooth")
def _crescent_grad(op, grad):
  dydx1, dydx2 = customtf.cb2_smooth_gradient(grad, op.inputs[0], op.inputs[1])
  return [dydx1, dydx2]  # List of one Tensor, since we have one input


@ops.RegisterGradient("Cb2")
def _crescent_grad(op, grad):
  dydx1, dydx2 = customtf.cb2_gradient(grad, op.inputs[0], op.inputs[1])
  return [dydx1, dydx2]  # List of one Tensor, since we have one input


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



# cb2
x1 = 2.0
x2 = 2.0
plrange = [-1,2]

def cb2(x1, x2):
    r1 = x1 * x1 + x2 * x2 * x2 * x2
    r2 = tf.pow(2. - x1, 2) + tf.pow(2. - x2, 2)
    r3 = 2. * tf.exp(x2 -x1)
    return tf.reduce_max([r1,r2,r3], axis=0)
  

smfactors= [10., 50., 100., 500., 1000., 5000., 10000., np.inf]
numits = [200, 300, 400, 500, 750, 1000, 1500, 2000]

table = np.empty(shape=[len(smfactors), len(numits)])

for i,smfactor in enumerate(smfactors):
    for j,numit in enumerate(numits):
        if smfactor == np.inf:
            path_smooth = get_optimization_path(x1, x2, numit, cb2, smfactor)
        else:
            path_smooth = get_optimization_path(x1, x2, numit, customtf.cb2_smooth, smfactor)
        table[i,j] = customtf.cb2(*path_smooth[-1])
        # path_smooth = get_optimization_path(x1, x2, numit, customtf.crescent_smooth, smfactor)
        # table[i,j] = customtf.crescent(*path_smooth[-1])


for i in range(table.shape[0]):
    print( [float('%.4f' % n) for n in table[i, :]] )
