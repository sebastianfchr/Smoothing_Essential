# import sys
# sys.path.append('../..')
# sys.path.append('../')
# from build.pybound.smfuncs import smooth_dijkstra, set_smoothing_factor
from utils.plotting_utils import *
import networkx as nx 
import matplotlib
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
import matplotlib.pyplot as plt
import itertools
from utils.comparison_utils import *

def make_edge_labels(mat):
    outDict = {}
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i,j] != 0: outDict[(i,j)] = str(round(mat[i,j], 2))
    return outDict

def make_edge_colors(edges, weights, m = None):
    vs = list(weights.values())
    if m == None: # no manually defined boundary
        m = np.max([np.abs(np.max(vs)), np.abs(np.min(vs))])
    # bare_edge_vals = np.reshape(mat, -1)
    f = lambda e: (0.1,0.1,e/m, max(0.1,e/m)) if e>=0 else (-e/m,0.1,0.1, max(0.1,-e/m))
    cols = [f(weights[edge]) for edge in edges] 
    return cols

def plot_complete(mat_curr, deriv_mat, seed=None, pos_dict=None, savepath=None, figsize=(10,5)):
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=figsize)
    # nx.write_latex(G, "/home/seb/Desktop/present/PhD_seminar_2/content/tikz/networkxgraphs/just_my_figure.tex", as_document=True, document_wrapper='{content}', default_edge_options="[->]")

    G_forward = nx.from_numpy_array(mat_curr, create_using=nx.DiGraph)
    G_deriv = nx.from_numpy_array(deriv_mat, create_using=nx.DiGraph)

    node_list = None
    if pos_dict != None:    # pos_dict overrules seed!
        node_list = list(range(mat_curr.shape[0]))
        pos = pos_dict
    elif seed != None:      # otherwise use seed
        pos = nx.spring_layout(G_forward, seed=seed)
    else:
        raise Exception("Either node_list with pos, or seed must be given")
    # print(pos_dict)
    # print(nodes)
    # exit(0)
    # extract weights for labels
    forward_edge_weights_dict = make_edge_labels(mat_curr)
    deriv_edge_weights_dict = make_edge_labels(deriv_mat)
    # colors based on edges 
    G_deriv_colors = make_edge_colors(G_deriv.edges, nx.get_edge_attributes(G_deriv, "weight"))

    for (ax, G, edge_dict, edgelabelcolor, edgecolors) in \
        zip([ax1, ax2], [G_forward, G_deriv], 
            [forward_edge_weights_dict, deriv_edge_weights_dict],
            ['black', 'gray'], 
            ["black", G_deriv_colors]
        ) :
        # =========== 1) draw for upper graph ==============
        Gbunch = nx.draw_networkx_nodes(
            G, pos=pos,
            nodelist = node_list,
            node_size=500, 
            node_color='tab:blue', 
            alpha=0.3,
            ax=ax
        )
        nx.draw_networkx_labels(
            G, pos, 
            labels={node: node for node in G.nodes()},
            ax=ax
        ) 
        nx.draw_networkx_edges(
            G, pos, 
            width=0.1,
            node_size=500,
            arrows = True,
            arrowstyle=matplotlib.patches.ArrowStyle("simple, head_length=1, head_width=.4"),
            ax=ax,
            edge_color=edgecolors,

        )
        nx.draw_networkx_edge_labels(
            G_forward, pos, 
            edge_labels=edge_dict,
            font_color=edgelabelcolor,
            ax=ax
        )
        ax.axis('off')
    plt.subplots_adjust(top=1, bottom=0, hspace=0.074)



def plot_backprop_stages(mat_curr, deriv_increments_per_run, paths, seed=5):
    """ plot each stage of the backpropagation """
    fig, [[ax1, ax2],[ax3,ax4]] = plt.subplots(2, 2, figsize=(10,5))
    # nx.write_latex(G, "/home/seb/Desktop/present/PhD_seminar_2/content/tikz/networkxgraphs/just_my_figure.tex", as_document=True, document_wrapper='{content}', default_edge_options="[->]")

    G_forward = nx.from_numpy_array(mat_curr, create_using=nx.DiGraph)
    pos = nx.spring_layout(G_forward, seed=seed)
    # pos = nx.spring_layout(G_forward, seed=2)


    G_forward_p1 = G_forward.subgraph(paths[0])
    G_deriv_p1 = nx.from_numpy_array(deriv_increments_per_run[0], create_using=nx.DiGraph)
    G_forward_p2 = G_forward.subgraph(paths[1])
    G_deriv_p2 = nx.from_numpy_array(deriv_increments_per_run[1], create_using=nx.DiGraph)


    # extract weights for labels
    forward_edge_weights_dict = make_edge_labels(mat_curr)
    deriv_edge_weights_dict_p1 = make_edge_labels(deriv_increments_per_run[0])
    deriv_edge_weights_dict_p2 = make_edge_labels(deriv_increments_per_run[1])
    # colors based on edges 

    allweights = list(nx.get_edge_attributes(G_deriv_p1, "weight").values()) + list(nx.get_edge_attributes(G_deriv_p2, "weight").values())
    m = max(abs(max(allweights)), abs(min(allweights)))
    G_deriv_p1_colors = make_edge_colors(G_deriv_p1.edges, nx.get_edge_attributes(G_deriv_p1, "weight"), m)
    G_deriv_p2_colors = make_edge_colors(G_deriv_p2.edges, nx.get_edge_attributes(G_deriv_p2, "weight"), m)

    for (ax, G, edge_dict, edgelabelcolor, edgecolors) in \
        zip([ax1, ax2, ax3, ax4], [G_forward_p1, G_deriv_p1, G_forward_p2,  G_deriv_p2], 
            [forward_edge_weights_dict, deriv_edge_weights_dict_p1, forward_edge_weights_dict, deriv_edge_weights_dict_p2],
            ['black', 'gray', 'black', 'gray'], 
            ["black", G_deriv_p1_colors, "black", G_deriv_p2_colors]
        ) :
        # =========== 1) draw for upper graph ==============
        Gbunch = nx.draw_networkx_nodes(
            G, pos,
            node_size=500, 
            node_color='tab:blue', 
            alpha=0.3,
            ax=ax
        )
        nx.draw_networkx_labels(
            G, pos, 
            labels={node: node for node in G.nodes()},
            ax=ax
        ) 
        nx.draw_networkx_edges(
            G, pos, 
            width=0.1,
            node_size=500,
            arrows = True,
            arrowstyle=matplotlib.patches.ArrowStyle("simple, head_length=1, head_width=.4"),
            ax=ax,
            edge_color=edgecolors,

        )
        nx.draw_networkx_edge_labels(
            G_forward, pos, 
            edge_labels=edge_dict,
            font_color=edgelabelcolor,
            ax=ax
        )
        ax.axis('off')
    plt.subplots_adjust(top=1, bottom=0, hspace=0.074)
    # plt.savefig('/home/seb/Desktop/present/PhD_seminar_2/content/graph_plots/small_graph_subgraphs_derivatives_improved.pgf')









def plot_backprop_stages_deriv(mat_curr, deriv_increments_per_run, paths, plot_forward_paths_only=False, seed=5):
    """ plot each stage of the backpropagation """

    assert(len(deriv_increments_per_run) == len(paths))
    num_plots = len(paths)

    fig = plt.figure(figsize=(10,10))
    
    closest_square = np.ceil(np.sqrt(num_plots)).astype(int)
    upper_begin = int(np.ceil((1/2 - 0.5*1/closest_square)*101))
    upper_end = int(np.ceil((1/2 + 0.5*1/closest_square)*101))

    pseudo_gs_top = matplotlib.gridspec.GridSpec(1+closest_square, 101)
    ax_top = plt.subplot(pseudo_gs_top[0,upper_begin:upper_end])
    ax_top.axis("off")

    gs = matplotlib.gridspec.GridSpec(1+closest_square, closest_square)
    # axes after first row in row-major
    axes_sequential = []
    for (i,j) in itertools.product(range(closest_square), range(closest_square)):
        ax = plt.subplot(gs[1+i,j])
        axes_sequential.append(ax)
        ax.axis("off")

    G_orig = nx.from_numpy_array(mat_curr, create_using=nx.DiGraph)
    pos = nx.spring_layout(G_orig, seed=seed)
    G_orig_edge_weights_dict = make_edge_labels(mat_curr)

    # draw main graph centrally 
    nx.draw_networkx_nodes(G_orig, pos, node_size=500, node_color='tab:blue', alpha=0.3, ax=ax_top)
    nx.draw_networkx_labels(G_orig, pos, labels={node: node for node in G_orig.nodes()},ax=ax_top) 
    nx.draw_networkx_edges(G_orig, pos, width=0.1, node_size=500, arrows = True, arrowstyle=matplotlib.patches.ArrowStyle("simple, head_length=1, head_width=.4"), ax=ax_top, edge_color="black",)
    nx.draw_networkx_edge_labels(G_orig, pos, edge_labels=G_orig_edge_weights_dict, font_color="black", ax=ax_top)


    if(not plot_forward_paths_only):     # plot the derivatives of each forward path
        # extract all weights of all graphs for colors!
        allweights = []
        for i in range(len(deriv_increments_per_run)):
            G_deriv_of_path_i = nx.from_numpy_array(deriv_increments_per_run[i], create_using=nx.DiGraph)
            allweights += list(nx.get_edge_attributes(G_deriv_of_path_i, "weight").values())
        m = max(abs(max(allweights)), abs(min(allweights)))  # graph colors will be formed based on [-m, m]!

        for i in range(num_plots):
            ax = axes_sequential[i]
            
            # derivative of given path
            G_deriv_of_path_i = nx.from_numpy_array(deriv_increments_per_run[i], create_using=nx.DiGraph)
            G_deriv_of_path_i_edge_weights_dict = make_edge_labels(deriv_increments_per_run[i])
            G_deriv_of_path_i_colors = make_edge_colors(G_deriv_of_path_i.edges, nx.get_edge_attributes(G_deriv_of_path_i, "weight"), m)

            # nodes and labels
            nx.draw_networkx_nodes(G_orig, pos, node_size=500, node_color='tab:blue', alpha=0.3, ax=ax)
            nx.draw_networkx_labels(G_orig, pos, labels={node: node for node in G_orig.nodes()},ax=ax) 

            nx.draw_networkx_edges(G_deriv_of_path_i, pos, width=0.1,node_size=500,arrows = True,
                arrowstyle=matplotlib.patches.ArrowStyle("simple, head_length=1, head_width=.4"),ax=ax, edge_color=G_deriv_of_path_i_colors,)
            nx.draw_networkx_edge_labels(G_orig, pos, edge_labels=G_deriv_of_path_i_edge_weights_dict,
                font_color="black", ax=ax)

    else: 
        for i in range(num_plots):
            ax = axes_sequential[i]
            G_orig_path_i = G_orig.subgraph(paths[i])
            
            visited_edges = extract_edge_sequence(paths[i]) 
            visited_edge_labels_dict = {key : G_orig_edge_weights_dict[key] for key in visited_edges}
            
            unvisited_edges = [e for e in G_orig.edges if e not in visited_edges]
            unvisited_edge_labels_dict = {key : G_orig_edge_weights_dict[key] for key in unvisited_edges}
            
            nx.draw_networkx_nodes(G_orig, pos, node_size=500, node_color='tab:blue', alpha=0.3, ax=ax)
            nx.draw_networkx_labels(G_orig, pos, labels={node: node for node in G_orig.nodes()},ax=ax) 
            # wisited !!
            nx.draw_networkx_edges(G_orig_path_i, pos, edgelist=visited_edges, width=0.1, node_size=500, arrows = True, arrowstyle=matplotlib.patches.ArrowStyle("simple, head_length=1, head_width=.4"), ax=ax, edge_color="black",)
            nx.draw_networkx_edge_labels(G_orig, pos, edge_labels=visited_edge_labels_dict, font_color="black", ax=ax)
            # unvisited !!
            nx.draw_networkx_edges(G_orig_path_i, pos, edgelist=unvisited_edges, width=0.1, node_size=500, arrows = True, arrowstyle=matplotlib.patches.ArrowStyle("simple, head_length=1, head_width=.4"), ax=ax, edge_color="black", alpha=0.2)
            nx.draw_networkx_edge_labels(G_orig, pos, edge_labels=unvisited_edge_labels_dict, font_color="gray", alpha=0.2, ax=ax)


    plt.subplots_adjust(top=1, bottom=0, hspace=0)
    plt.show()
    # plt.savefig('/home/seb/Desktop/present/PhD_seminar_2/content/graph_plots/small_graph_subgraphs_derivatives_improved.pgf')



