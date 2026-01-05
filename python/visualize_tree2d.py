import os
import numpy as np
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from likd_tree import KDTree, get_tree_snapshot


def idx_to_color(idx):
    # map integer idx to an HSV hue and convert to hex; deterministic and spaced
    h = (idx * 137) % 360  # golden-angle-like spacing
    s = 0.65
    v = 0.85
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h/360.0, s, v)
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))


def layout_tree_simple(snap):
    # Create a simple depth/inorder layout similar to Reingold-Tilford
    children = {n['idx']: [] for n in snap}
    parent = {}
    for n in snap:
        if n['parent'] != -1:
            children[n['parent']].append(n['idx'])
            parent[n['idx']] = n['parent']

    roots = [n['idx'] for n in snap if n['parent'] == -1]
    positions = {}
    depths = {}
    x_counter = 0

    def dfs(u, depth):
        nonlocal x_counter
        depths[u] = depth
        ch = children.get(u, [])
        if not ch:
            # leaf
            positions[u] = (x_counter, -depth)
            x_counter += 1
            return positions[u][0]
        else:
            child_xs = []
            for c in ch:
                cx = dfs(c, depth + 1)
                child_xs.append(cx)
            # center parent x between children (average)
            px = sum(child_xs) / len(child_xs)
            positions[u] = (px, -depth)
            return px

    for r in roots:
        dfs(r, 0)
    return positions


def collect_snapshots(tree, pts, batch=2):
    snapshots = []
    tree.build(pts[:4])
    snapshots.append(get_tree_snapshot(tree))
    for i in range(4, len(pts), batch):
        tree.add_points(pts[i:i+batch])
        tree.wait_for_rebuild()
        snapshots.append(get_tree_snapshot(tree))
    return snapshots


def build_plotly_frame_from_snapshot(snap):
    pos = layout_tree_simple(snap)
    node_ids = sorted(pos.keys())

    Xn = [pos[n][0] for n in node_ids]
    Yn = [pos[n][1] for n in node_ids]
    labels = [str(n) for n in node_ids]

    # edges
    Xe = []
    Ye = []
    for n in snap:
        if n['parent'] != -1:
            a = n['parent']
            b = n['idx']
            xa, ya = pos[a]
            xb, yb = pos[b]
            Xe += [xa, xb, None]
            Ye += [ya, yb, None]

    edge_trace = go.Scatter(x=Xe, y=Ye, mode='lines', line=dict(color='rgb(210,210,210)', width=1), hoverinfo='none')
    # assign a deterministic color per node id
    node_colors = [idx_to_color(n) for n in node_ids]
    node_trace = go.Scatter(x=Xn, y=Yn, mode='markers', marker=dict(symbol='circle-dot', size=14, color=node_colors, line=dict(color='rgb(50,50,50)', width=1)), text=labels, hoverinfo='text')

    # no new-node highlighting here; it's added when building frames where previous frame is known

    annotations = []
    for k, nid in enumerate(node_ids):
        annotations.append(dict(text=str(nid), x=Xn[k], y=Yn[k], xref='x', yref='y', showarrow=False, font=dict(color='rgb(250,250,250)', size=10)))

    return edge_trace, node_trace, annotations


def build_plotly_animation(snapshots, out_html='tree_plotly.html'):
    frames = []
    prev_nodes = set()
    for i, snap in enumerate(snapshots):
        edge_trace, node_trace, annotations = build_plotly_frame_from_snapshot(snap)
        cur_nodes = set(n['idx'] for n in snap)
        new_nodes = sorted(list(cur_nodes - prev_nodes))
        new_trace = None
        if new_nodes:
            # compute positions for new nodes
            pos = layout_tree_simple(snap)
            x_new = [pos[n][0] for n in new_nodes]
            y_new = [pos[n][1] for n in new_nodes]
            new_colors = [idx_to_color(n) for n in new_nodes]
            # brighter color and star marker to make highlights obvious
            new_trace = go.Scatter(x=x_new, y=y_new, mode='markers', marker=dict(symbol='star', size=30, color=new_colors, line=dict(color='rgb(0,0,0)', width=3)), hoverinfo='text')

        data_traces = [edge_trace, node_trace]
        if new_trace is not None:
            data_traces.append(new_trace)

        frames.append(dict(name=str(i), data=data_traces, layout=dict(title_text=f'Frame {i} - {len(snap)} nodes', annotations=annotations)))
        prev_nodes = cur_nodes

    if not frames:
        print('No frames to render')
        return

    init_traces = frames[0]['data']
    fig = go.Figure(data=init_traces,
                    layout=go.Layout(title=frames[0]['layout']['title_text'], showlegend=False, xaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False), yaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False), margin=dict(l=40, r=40, b=85, t=100), plot_bgcolor='rgb(248,248,248)', annotations=frames[0]['layout'].get('annotations', []), updatemenus=[dict(type='buttons', showactive=False, y=1, x=1.1, xanchor='right', yanchor='top', pad=dict(t=0, r=10), buttons=[dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=800, redraw=True), fromcurrent=True)])])]),
                    frames=[go.Frame(data=f['data'], name=f['name'], layout=go.Layout(title_text=f['layout']['title_text'], annotations=f['layout'].get('annotations', []))) for f in frames]
                   )

    fig.write_html(out_html, auto_open=False)
    print('Wrote', out_html)


def main():
    np.random.seed(42)
    n_points = 40
    pts = np.random.randn(n_points, 3).astype(np.float32)

    tree = KDTree()
    snapshots = collect_snapshots(tree, pts, batch=2)

    build_plotly_animation(snapshots, out_html='tree_plotly.html')


if __name__ == '__main__':
    main()
