"""
tree_visualizer.py
基于 NetworkX 与 Plotly 的 N叉树拓扑网络交互渲染器
(已彻底移除 C++ Graphviz 底层依赖与 Plotly 弃用参数)
"""
import networkx as nx
import plotly.graph_objects as go
import numpy as np

class TreeVisualizer:
    def __init__(self, nodes_list: list):
        self.nodes = nodes_list
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        """ 遍历所有像源，构建有向图 (Directed Graph) """
        for n in self.nodes:
            # 添加节点，注入悬停属性
            self.graph.add_node(
                n.node_id, 
                z=n.z, 
                q=n.q, 
                generation=n.generation,
                b_id=n.gen_by_boundary
            )
            # 添加有向边
            if n.parent_id is not None:
                self.graph.add_edge(n.parent_id, n.node_id)

    def generate_plotly_figure(self):
        """ 利用纯 Python 原生算法生成树状布局 """
        if len(self.nodes) == 0:
            return go.Figure()

        # 【核心修正 1】：彻底抛弃 pygraphviz，使用 NetworkX 内置的多分部布局
        # 完美利用我们底层的 generation 层级属性实现分层
        pos = nx.multipartite_layout(self.graph, subset_key='generation')
        
        # 将横向生成的树转置为从上到下的竖向树 (y为层级负值，x为横向排布)
        pos = {node: (coords[1], -coords[0]) for node, coords in pos.items()}
        
        edge_x =[]
        edge_y = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # 连线轨迹
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        hover_texts = []
        node_colors =[]

        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            data = self.graph.nodes[node]
            # 组织强悍的悬停信息面板
            text = (f"<b>节点 ID:</b> {node}<br>"
                    f"<b>层级 (Gen):</b> {data['generation']}<br>"
                    f"<b>复坐标 Z:</b> {np.real(data['z']):.4f} + {np.imag(data['z']):.4f}j<br>"
                    f"<b>强度 Q:</b> {data['q']:e}<br>"
                    f"<b>反演边界:</b> Γ_{data['b_id']}")
            hover_texts.append(text)
            # 按层级赋色
            node_colors.append(data['generation'])

        # 节点散点
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=hover_texts,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                reversescale=True,
                color=node_colors,
                size=12,
                # 【核心修正 2】：剔除引发 ValueError 的 xanchor 和 titleside 扁平参数
                colorbar=dict(thickness=15, title='Generation'),
                line_width=1.5
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='N-ary Tree Topology (Interactive)',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        return fig
