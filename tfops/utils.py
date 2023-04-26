import numpy as np


class meshtype():
    """ Shorthand for mesh-structure that can be accessed as doublegrid or sequential """

    def __init__(self, x1range, x2range, resolution_per_dimension):
        x1s = np.linspace(*x1range, resolution_per_dimension)
        self.n = len(x1s)
        x2s = np.linspace(*x2range, resolution_per_dimension)
        self.m = len(x2s)
        # ((n, m), (n, m)) => (2, n, m)
        self.doublegrid = np.stack(np.meshgrid(x1s, x2s, indexing='ij'))

    def as_doublegrid(self):
        return self.doublegrid  # (2, n, m)

    def as_sequential(self):
        # => (2, n, m) => (2, n*m) => (n*m, 2)
        return np.transpose(np.reshape(self.doublegrid, (2, -1)))

    def plot_dims(self):
        return (self.n, self.m)


def plot_multi_discontinuous(ax, xbounds, xfs, function, num_points=100, *args, **kwargs):
    """ 
        for each xf in xfs:
        function is interrupted at xf. Plot separately for [xbounds[0], floor(x_f)], [ceil(x_f), xbounds[1]] 
        Here, ceil and floor are in terms of discretized intervals
    """

    remlabel = kwargs.pop('label') if 'label' in kwargs else None

    linesCoords = []
    xfs = np.sort(xfs)
    old_interval_end = xbounds[0]
    xs = np.linspace(*xbounds, num_points)
    for i, xf in enumerate(xfs):
        # identify uneven point by calculating what np's spacings are, and
        np_interval_width = (xbounds[1]-xbounds[0])/(num_points-1)
        intervals_to_boundary = int((xf-xbounds[0])//np_interval_width)
        xs_curr = xs[old_interval_end:intervals_to_boundary+1]
        old_interval_end = intervals_to_boundary+1
        ys_curr = function(xs_curr)

        if (i>0): kwargs['color'] = lline.get_color()
        # color = kwargs['color'] if i==0 else lline.get_color()
        # kwargs['color'] = kwargs['color'] if kwargs['color'] or i==0 else lline.get_color()

        # NOTE: kwargs only once (stupid hack for label)!
        lline, = ax.plot(xs_curr, ys_curr, *args, **kwargs)
        linesCoords.append((xs_curr, ys_curr))

    xs_fin = xs[old_interval_end:]
    ys_fin = function(xs_fin)
    lline, ax.plot(xs_fin, ys_fin, *args, label=remlabel, **kwargs)
    linesCoords.append((xs_fin, ys_fin))
    return linesCoords


def run_RnR1_on_mesh(RnR1function, mesh_for_f):
    # runs f(x1, x2, .. x_n) R^n->R^1 on sequential mesh, then reshapes output to original form
    # sequential mesh: [[x1_1, x2_1, ..., x_n1], [x1_2, x2_2, ..., x_n2] ...] => [y1, y2 ...]
    args_per_listentry = mesh_for_f.as_sequential()
    ys_sequential = list(map(lambda x: RnR1function(*x), args_per_listentry))
    # apply filter!# ===================

    # reshape to original
    return np.reshape(ys_sequential, mesh_for_f.plot_dims())


def run_RnR1_on_mesh_discontinuous(RnR1function, mesh_for_f, maskfunc = None):
    # runs f(x1, x2, .. x_n) R^n->R^1 on sequential mesh, then reshapes output to original form
    # sequential mesh: [[x1_1, x2_1, ..., x_n1], [x1_2, x2_2, ..., x_n2] ...] => [y1, y2 ...]
    args_per_listentry = mesh_for_f.as_sequential()
    ys_sequential = list(map(lambda x: RnR1function(*x), args_per_listentry))
    # apply filter!# ===================
    
    if(maskfunc):
        ys_sequential = np.stack([y if maskfunc(*xs) else np.nan for (xs, y) in zip(args_per_listentry, ys_sequential) ])

    # reshape to original
    return np.reshape(ys_sequential, mesh_for_f.plot_dims())


def run_RnR2_on_mesh(RnR2function, mesh_for_f):
    # runs f(x1, x2, .. x_n), R^n->R^2 on sequential mesh, then reshapes output to original form
    # sequential mesh: [[x1_1, x2_1, ..., x_n1], [x1_2, x2_2, ..., x_n2] ...] => [[y1_1, y1_2], [y2_1, y2_2] ...]
    args_per_listentry = mesh_for_f.as_sequential()
    y1s_ys_2_sequential = np.stack(list(map(lambda x: RnR2function(*x), args_per_listentry)))

    
    # apply filter!# ===================

    # reshape to original
    return np.reshape(y1s_ys_2_sequential[:, 0], mesh_for_f.plot_dims()), np.reshape(y1s_ys_2_sequential[:, 1], mesh_for_f.plot_dims())
