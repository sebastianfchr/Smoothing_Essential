import numpy as np


class meshtype():
    """ Shorthand for mesh-structure that can be accessed as doublegrid or sequential """

    def __init__(self, x1range, x2range, resolution_per_dimension):
        x1s = np.linspace(*x1range, resolution_per_dimension)
        self.m = len(x1s)
        x2s = np.linspace(*x2range, resolution_per_dimension)
        self.n = len(x2s)
        # ((m, n), (m, n)) => (2, m, n)
        self.doublegrid = np.stack(np.meshgrid(x1s, x2s, indexing='ij'))
        # print(len(x1s))
        # print(len(x2s))
        # print(self.doublegrid.shape)

        
    def as_doublegrid(self):
        return self.doublegrid  # (2, m, n)

    def as_sequential(self):

        # => (2, m, n) => (2, m*n) => (m*n, 2)
        return np.transpose(np.reshape(self.doublegrid, (2, -1)))

    def plot_dims(self):
        return (self.m, self.n)


def run_RnR1_on_mesh(RnR1function, mesh_for_f):
    # run f(x1, x2, .. x_n) R^n->R^1 on sequential mesh, then reshape output to original form

    # sequential mesh: [[x1_1, x2_1, ..., x_n1], [x1_2, x2_2, ..., x_n2] ...] => [y1, y2 ...]
    mesh_sequential = mesh_for_f.as_sequential()

    ys_sequential = np.stack(list(map(lambda x: RnR1function(*x), mesh_sequential)))
    # reshape to original
    # mesh_sequential has come from (2, m, n) => (2,n*m) ==transpose=> (m*n, 2)
    # so ys_sequential must be from (m*n, K) ==transpose==> (K, m*n) => (K, m, n)
    print(ys_sequential.shape)
    ys = np.transpose(ys_sequential, (1,0)) # (m*n, K) =>(K, m*n)
    print(ys.shape)
    ys = np.reshape(ys, (-1, *mesh_for_f.plot_dims()))

    return ys #np.reshape(ys_sequential, (-1, *mesh_for_f.plot_dims()))


def run_RnR2_on_mesh(RnR2function, mesh_for_f):
    # run f(x1, x2, .. x_n), R^n->R^2 on sequential mesh, then reshape output to original form
    # sequential mesh: [[x1_1, x2_1, ..., x_n1], [x1_2, x2_2, ..., x_n2] ...] => [[y1_1, y1_2], [y2_1, y2_2] ...]
    args_per_listentry = mesh_for_f.as_sequential()

    y1s_ys_2_sequential = np.stack(
        list(map(lambda x: RnR2function(*x), args_per_listentry)))

    # reshape to original
    return np.reshape(y1s_ys_2_sequential[:, 0], mesh_for_f.plot_dims()), np.reshape(y1s_ys_2_sequential[:, 1], mesh_for_f.plot_dims())
