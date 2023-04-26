import matplotlib.pyplot as plt
import numpy as np
from utils import plot_multi_discontinuous
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': "15",
})


def plot_names_on_lines(ax, lines_coords, names=[r'$f_1 \circ g_1$', r'$f_2 \circ g_1$', r'$f_1 \circ g_2$']):
# ============= 2nd plot: concatenated artifacts, both, without artifacts =========
    for (line_xs, line_ys), fname in zip(lines_coords, names):
        # take tangent from midpoint, rotate 90 degrees
        mid = len(line_xs)//2
        mid_tangent = (line_ys[mid+1]-line_ys[mid-1]) / \
            (line_xs[mid+1]-line_xs[mid-1])
        mid_tangent_vector = np.stack([[1.], [mid_tangent]])
        theta = np.pi/2  # 90 degrees
        rotated_vector = [[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]] @ mid_tangent_vector
        # scale
        rotated_vector /= np.sqrt(rotated_vector[0]**2+rotated_vector[1]**2)
        rotated_vector *= 0.2
        # plot the text at mid+vector
        ax.text(line_xs[mid]+rotated_vector[0]-0.2,
                line_ys[mid]+rotated_vector[1], fname, color='gray')




def make_sigmoids(b, h=10):

    def right_sigmoid(x, h=h):
        return 1./(1.+np.exp(-(x-b)*h))

    def left_sigmoid(x, h=h):
        return 1-right_sigmoid(x, h)

    return left_sigmoid, right_sigmoid 

def f1(x): return -2*x+1  # or PlainFunction(...) possible!
def f2(x): return x
def g1(x): return x+1
def g2(x): return -x+2

sig_xf_l, sig_xf_r = make_sigmoids(2)
sig_xg_l, sig_xg_r = make_sigmoids(2)


def f(x):
    if x < 2: return f1(x)
    else: return f2(x)

def g(x):
    if x < 2: return g1(x)
    else: return g2(x)

def f_sm(x):
    return sig_xf_l(x)* f1(x) + sig_xf_r(x)* f2(x)

def g_sm(x):
    return sig_xg_l(x)* g1(x) + sig_xg_r(x)* g2(x)


# contributions clean
def c11(x): return sig_xf_l(g1(x)) * sig_xg_l(x)
def c12(x): return sig_xf_l(g2(x)) * sig_xg_r(x) 
def c21(x): return sig_xf_r(g1(x)) * sig_xg_l(x) 
def c22(x): return sig_xf_r(g2(x)) * sig_xg_r(x)

# contributions naive
def nc11(x): return sig_xf_l(g_sm(x)) * sig_xg_l(x)
def nc12(x): return sig_xf_l(g_sm(x)) * sig_xg_r(x)
def nc21(x): return sig_xf_r(g_sm(x)) * sig_xg_l(x)
def nc22(x): return sig_xf_r(g_sm(x)) * sig_xg_r(x)

def f_o_g(x): return f(g(x))
def f_o_g_np(xs): return np.stack(map(lambda x: f_o_g(x), xs))
def f_sm_o_g_sm(xs): return f_sm(g_sm(xs))
def tilde_f_o_g(xs): return c11(xs) * f1(g1(xs)) + c12(xs) * f1(g2(xs)) + c21(xs) * f2(g1(xs)) + c22(xs) * f2(g2(xs))

# s2, s1 = make_sigmoids(2)
# qf1_cont, qf2_cont = lambda x: s1(q(x)), lambda x: s2(q(x))
# gf1_cont, gf2_cont = lambda x: s1(g(x)), lambda x: s2(g(x))



xbounds = [0,3]
# xs = np.linspace(*xbounds, 101)
xs = np.linspace(*xbounds, 4000)
sig, sigc = make_sigmoids(0.4)

fig = plt.figure(figsize=(13, 5))

# =============================
ax1 = fig.add_subplot(221)
poss_left = plot_multi_discontinuous(ax1, xbounds, [1,2], f_o_g_np, num_points=500, label=r'$f\circ g$', color='black')
plot_names_on_lines(ax1, poss_left)
ax1.plot(xs, tilde_f_o_g(xs), color = u'gray', alpha = 0.9, label=r"$\widetilde{f\circ g}$")
ax1.set_yticks([-2.5, 0, 2.5, 5])

plt.setp(ax1.get_xticklabels(), visible=False)
# =============================
ax3 = fig.add_subplot(223)
ax3.set_yticks([0, 0.5, 1])
# ax3.plot([], [], linestyle=None, label=r'$p(f\circ g = ...)$')
ax3.plot(xs, c11(xs), label=r'$f_1 \circ g_1$', alpha=0.7, color='black', linestyle='-.')
ax3.plot(xs, c12(xs), label=r'$f_1 \circ g_2$', alpha=1., color='gray', linestyle='dotted')
ax3.plot(xs, c21(xs), label=r'$f_2 \circ g_1$', alpha=0.5, color='black', linestyle='--')
ax3.plot(xs, c22(xs), label=r'$f_2 \circ g_2$', alpha=0.9, color='black')

# =============================
ax2 = fig.add_subplot(222, sharey = ax1)
poss_right =  plot_multi_discontinuous(ax2, xbounds, [1,2], f_o_g_np, num_points=500, label=r'$f\circ g$', color='black')
plot_names_on_lines(ax2, poss_right)
ax2.plot(xs, f_sm(g_sm(xs)), color = u'gray', alpha = 0.9, label=r'$\tilde{f}\circ \tilde{g}$')
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)
# =============================
ax4 = fig.add_subplot(224, sharex = ax2)
ax4.plot(xs, nc11(xs), label=r'$f_1 \circ g_1$', alpha=0.7, color='black', linestyle='-.')
ax4.plot(xs, nc12(xs), label=r'$f_1 \circ g_2$', alpha=1., color='gray', linestyle='dotted')
ax4.plot(xs, nc21(xs), label=r'$f_2 \circ g_1$', alpha=0.5, color='black', linestyle='--')
ax4.plot(xs, nc22(xs), label=r'$f_2 \circ g_2$', alpha=0.9, color='black')
plt.setp(ax4.get_yticklabels(), visible=False)

ax3.set_xticklabels([])
ax4.set_xticklabels([])

l = ax3.legend(); l.set_title(r"$p(\widetilde{f\circ g} = \dots)$")
l = ax4.legend(); l.set_title(r"$p(\tilde{f}\circ \tilde{g} = \dots)$")
ax1.legend()
ax2.legend()

for ax in [ax1, ax2, ax3, ax4]:
    ax.axvline(x=1, linestyle='dotted', color='gray', alpha=0.5)
    ax.axvline(x=2, linestyle='dotted', color='gray', alpha=0.5)



fig.subplots_adjust(bottom=0.025, top=0.98, left=0.03, right=0.99, wspace=0.03, hspace=0.03)

plt.show()
plt.savefig("/home/seb/Desktop/write/smoothing/content/figures/artefact_vs_proper_bw.pgf")
# plot_or_save('smoothing_artefact.pgf', False)