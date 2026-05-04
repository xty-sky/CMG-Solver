"""
app.py
多边界广义镜像阵列法 半数值解析求解系统
包含：图层拦截防穿透、点击联动高亮、严格序号标记
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go


class TrivialSolutionException(Exception):
    """触发极值定理退化条件时抛出的自定义异常"""
    pass

class SourceNode:
    def __init__(self, node_id, z, q_topo, parent_id=None, gen_by_boundary=None, generation=0, root_id=None, idx_in_gen=1):
        self.id = node_id
        self.z = z
        self.q_topo = q_topo  
        self.parent_id = parent_id
        self.gen_by = gen_by_boundary
        self.gen = generation
        self.root_id = root_id
        self.idx = idx_in_gen # 记录：这一代第几个节点

    def get_topological_potential(self, z_eval):
        distance = np.maximum(np.abs(z_eval - self.z), 1e-12)
        return - self.q_topo * np.log(distance)

class CircularBoundary:
    def __init__(self, b_id, center, radius, potential, K_mult):
        self.id = b_id
        self.c = center
        self.r = radius
        self.u = potential
        self.K = K_mult

    def reflect(self, source: SourceNode, base_new_id: str, start_idx: int) -> list:
        """
        映射裂变
        根据偏微分方程边界推演，一次跨界反射必然裂变为两个像源：
        1. 反演点像源 (T_i)：强度衰减 K_m
        2. 轴心点像源 (C_i)：强度衰减 -K_m，位置在圆心
        """
        nodes =[]
        
        # 1. 反演点像源 T_i
        z_inv = self.c + (self.r**2) / np.conj(source.z - self.c)
        q_inv = source.q_topo * self.K
        node_inv = SourceNode(f"{base_new_id}_T", z_inv, q_inv, source.id, self.id, source.gen + 1, source.root_id, start_idx)
        nodes.append(node_inv)
        
        # 2. 轴心点像源 C_i
        z_cen = self.c
        q_cen = - source.q_topo * self.K
        node_cen = SourceNode(f"{base_new_id}_C", z_cen, q_cen, source.id, self.id, source.gen + 1, source.root_id, start_idx + 1)
        nodes.append(node_cen)
        
        return nodes
    
class ImageTreeEngine:
    def __init__(self, boundaries, tolerance=1e-3):
        self.boundaries = boundaries
        self.tol = tolerance
        self.layers =[]
        self.node_cnt = 0

    def check_a_priori_conditions(self):
        if not self.boundaries: return
        u_vals = [b.u for b in self.boundaries]
        # 若所有边界电位极差趋于0，触发极值定理熔断
        if max(u_vals) - min(u_vals) < 1e-12:
            raise TrivialSolutionException(f"退化熔断: 边界电位全为 {u_vals[0]}V，域内无源，全域电势为常数。")

    def seed_roots(self, external_sources=None):
        """ 双模态场源播种 (Seeding) """
        layer_0 =[]
        idx = 1
        
        # 模态 II：各个边界包围的已知源强（稳磁场中为电流），在几何轴心自动播种
        for b in self.boundaries:
            if b.u != 0: # 若边界电位为0则无须播种
                root_id = f"root_B{b.id}"
                node = SourceNode(root_id, b.c, b.u, gen_by_boundary=b.id, root_id=root_id, idx_in_gen=idx)
                layer_0.append(node)
                self.node_cnt += 1
                idx += 1
                
        # 模态 I：可选的外部自由线电流场源
        if external_sources:
            for i, ext in enumerate(external_sources):
                root_id = f"root_Ext{i}"
                # gen_by_boundary设为-1，代表不属于任何物理边界的自由独立奇点
                node = SourceNode(root_id, ext['z'], ext['I'], gen_by_boundary=-1, root_id=root_id, idx_in_gen=idx)
                layer_0.append(node)
                self.node_cnt += 1
                idx += 1
                
        self.layers.append(layer_0)
    def seed_custom(self, z_eval: complex, q_real: float):
        """
        模态 I 自由播种：在用户指定域内坐标放入真实物理种子。
        注：播种点不能处于任何边界内部。
        """
        for b in self.boundaries:
            if abs(z_eval - b.c) <= b.r:
                raise ValueError("播种失败：源点 Z0 位于实心边界内部，不符合拉普拉斯物理求解域约束！")
                
        # 初始种子不再绑定单一边界，它将同时对所有边界引发第一代反演
        node = SourceNode(node_id="root_custom", z=z_eval, q_topo=q_real, gen_by_boundary=None, root_id="root_custom")
        self.layers.append([node])
        self.node_cnt += 1
    
    def step_forward(self):
        """ 执行单次广度优先搜索 (BFS)，触发映射裂变 """
        if not self.layers: return True
        curr_nodes = self.layers[-1]
        next_layer =[]
        idx = 1
        
        for node in curr_nodes:
            for b in self.boundaries:
                # 防御机制：严禁像源向刚刚生成它的边界立即反演 (防止死循环)
                if node.gen_by == b.id: continue 
                
                # 映射裂变接收：一个父节点生成两个子节点 (T 和 C)
                fission_nodes = b.reflect(node, f"n_{self.node_cnt}", idx)
                next_layer.extend(fission_nodes)
                
                self.node_cnt += 2
                idx += 2
                
        if not next_layer: return True # 无新节点生成
        self.layers.append(next_layer)
        return self._check_convergence()

    def _check_convergence(self):
        """ 严格基于等比余项误差评估 """
        # 由于引入了轴心补偿电荷，奇偶层的电量分布特性不同。
        # 必须至少迭代 3 层，才能建立跨越单周期的稳定公比评估
        if len(self.layers) < 3: return False
        
        # 计算每一层电流的绝对值平均衰减
        curr_q = np.mean([abs(n.q_topo) for n in self.layers[-1]])
        prev_q = np.mean([abs(n.q_topo) for n in self.layers[-2]])
        rho = curr_q / prev_q if prev_q > 0 else 0
        
        # 若处于发散域 (如纯静电 K=-1)，关闭余项判定，交由 GUI 强制上限截断
        if rho >= 0.99: return False 
        
        # 利用等比数列的无穷余项公式 E_N <= M * \rho / (1 - \rho) 进行严格放缩
        max_q = np.max([abs(n.q_topo) for n in self.layers[-1]])
        error_bound = max_q * rho / (1.0 - rho)
        
        return error_bound <= self.tol

    def get_nodes(self):
        return[n for layer in self.layers for n in layer]


class FieldSolver:
    def __init__(self, boundaries, nodes):
        self.boundaries = boundaries
        self.nodes = nodes
        self.lambdas = {n.root_id: 1.0 for n in nodes} 

    def solve_physics(self):
        """ 稳磁场特化：真实源电流即物理基底，直接旁路矩阵解算器。 """
        pass

    def evaluate_grid(self, x_range, y_range):
        """ 极简解析场重构，完全保留空间对数场的真实起伏 """
        X, Y = np.meshgrid(np.linspace(*x_range, 250), np.linspace(*y_range, 250))
        Z_grid = X + 1j * Y
        Az_grid = np.zeros_like(Z_grid, dtype=float)
        
        # 纯顺向累加真实格林函数
        for n in self.nodes:
            phys_I = n.q_topo * self.lambdas[n.root_id]
            # 引入 eps 防止 \ln(0) 引发底层 C 引擎 NaN 崩溃
            dist = np.maximum(np.abs(Z_grid - n.z), 1e-10)
            # 矢量磁势 A_z \propto -I \ln(r)
            Az_grid += -phys_I * np.log(dist)

        return X, Y, Az_grid

def plot_tree(nodes):
    """ 树状图渲染，注入 customdata 以备跨组件联动提取 """
    layers = {}
    for n in nodes:
        layers.setdefault(n.gen,[]).append(n)
        
    pos, x_coords, y_coords, texts, colors, custom_ids = {}, [], [], [], [],[]
    for gen, gen_nodes in layers.items():
        # y_vals 从上往下排列，刚好对应 idx_in_gen
        y_vals = np.linspace(1, -1, len(gen_nodes)+2)[1:-1] if len(gen_nodes)>1 else [0]
        for i, n in enumerate(gen_nodes):
            pos[n.id] = (gen, y_vals[i])
            x_coords.append(gen)
            y_coords.append(y_vals[i])
            colors.append(gen)
            custom_ids.append(n.id) 
            texts.append(f"Gen {n.gen} (第 {n.idx} 个)<br>Z: {np.real(n.z):.2f}+{np.imag(n.z):.2f}j<br>Q_topo: {n.q_topo:.2e}")
            
    edge_x, edge_y = [],[]
    for n in nodes:
        if n.parent_id:
            x0, y0 = pos[n.parent_id]
            x1, y1 = pos[n.id]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='#888', width=1), hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords, mode='markers', 
        customdata=custom_ids, 
        hovertext=texts, hoverinfo='text',
        marker=dict(size=12, color=colors, colorscale='Viridis', showscale=True, colorbar=dict(title='Generation'))
    ))
    fig.update_layout(title="N-ary Tree 繁衍图谱 (点击节点在左图亮显)", showlegend=False, 
                      xaxis=dict(title="迭代代数 (Generation)", tick0=0, dtick=1), 
                      yaxis=dict(showticklabels=False), margin=dict(l=10, r=10, t=40, b=10),
                      dragmode='pan') 
    return fig


st.set_page_config(page_title="多边界镜像法求解器", layout="wide")
st.title("⚡ 广义镜像法半解析全链路求解系统")

if 'engine' not in st.session_state: st.session_state.engine = None
if 'converged' not in st.session_state:
    st.session_state.converged = False
if 'selected_nodes' not in st.session_state:
    st.session_state.selected_nodes =[] # 初始化 FIFO 历史节点队列
if 'b_configs' not in st.session_state:
    st.session_state.b_configs =[
        {"id": 1, "u": 100.0, "r": 1.5, "x": -3.0, "y": 0.0},
        {"id": 2, "u": -50.0, "r": 1.0, "x": 3.0, "y": 0.0},
        {"id": 3, "u": 0.0, "r": 0.8, "x": 0.0, "y": 3.5}  # 恢复并默认开启第三边界
    ]

with st.sidebar:
    st.header("⚙️ 稳磁场边界与场源配置")
    
    # 动态边界参数
    for i, cfg in enumerate(st.session_state.b_configs):
        st.markdown(f"**边界 {cfg['id']} (磁介质圆柱)**")
        c1, c2, c3, c4 = st.columns(4)
        cfg['u'] = c1.number_input("内含电流 I", value=cfg.get('u', 10.0), step=10.0, key=f"I_{i}")
        cfg['r'] = c2.number_input("半径 R", value=cfg['r'], min_value=0.1, step=0.1, key=f"r_{i}")
        cfg['x'] = c3.number_input("圆心 x", value=cfg['x'], step=0.5, key=f"x_{i}")
        cfg['y'] = c4.number_input("圆心 y", value=cfg['y'], step=0.5, key=f"y_{i}")
        st.markdown("---")
        
    st.markdown("**外部自由场源 (可选播种)**")
    use_ext = st.checkbox("启用外部线电流源")
    if use_ext:
        e1, e2, e3 = st.columns(3)
        ext_I = e1.number_input("外部电流 I", value=-50.0, step=10.0)
        ext_x = e2.number_input("坐标 x", value=0.0, step=0.5)
        ext_y = e3.number_input("坐标 y", value=-2.0, step=0.5)
        
    k_m = st.slider("反射算子 (Km)", 0.1, 0.99, 0.85, 0.01)
    
    if st.button("🔄 播种并初始化 (Reset)", use_container_width=True):
        st.session_state.boundaries =[]
        for cfg in st.session_state.b_configs:
            # 此处 cfg['u'] 实际存储的是电流 I
            b = CircularBoundary(cfg['id'], complex(cfg['x'], cfg['y']), cfg['r'], cfg['u'], k_m)
            st.session_state.boundaries.append(b)
            
        st.session_state.engine = ImageTreeEngine(st.session_state.boundaries, tolerance=1e-4)
        
        ext_src = [{'z': complex(ext_x, ext_y), 'I': ext_I}] if use_ext else None
        st.session_state.engine.seed_roots(ext_src)
        
        st.session_state.converged = False
        st.session_state.selected_nodes =[]

    col1, col2 = st.columns(2)
    if col1.button("▶️ 单步繁衍"):
        if st.session_state.engine:
            st.session_state.converged = st.session_state.engine.step_forward()
    if col2.button("⏭️ 直接收敛"):
        if st.session_state.engine:
            while not st.session_state.engine.step_forward() and st.session_state.engine.node_cnt < 10000: 
                pass
            st.session_state.converged = True
            
    if st.button("🧹 清除高亮轨迹", use_container_width=True):
        st.session_state.selected_nodes =[]        


if st.session_state.engine:
    nodes = st.session_state.engine.get_nodes()
    
    # 状态回显
    st.info(f"**算法监控** | 当前迭代深度: `{st.session_state.engine.layers[-1][0].gen}` | 像源总数 (呈2^N爆炸): `{len(nodes)}` | 收敛状态: `{'✅已截断' if st.session_state.converged else '⏳演化中'}`")

    c1, c2 = st.columns([1.2, 1])
    
    with c2:
        fig_tree = plot_tree(nodes)
        selection_event = st.plotly_chart(fig_tree, use_container_width=True, on_select="rerun", selection_mode="points", key="tree_click")
                
        selected_node_id = None
        if selection_event and hasattr(selection_event.selection, "points"):
            points = selection_event.selection.points
            if len(points) > 0:
                pt = points[0]
                selected_node_id = pt.get("customdata") if isinstance(pt, dict) else getattr(pt, "customdata", None)
                if selected_node_id:
                    if selected_node_id in st.session_state.selected_nodes:
                        st.session_state.selected_nodes.remove(selected_node_id)
                    st.session_state.selected_nodes.append(selected_node_id)
                    if len(st.session_state.selected_nodes) > 5:
                        st.session_state.selected_nodes.pop(0)
    
    with c1:
        solver = FieldSolver(st.session_state.engine.boundaries, nodes)
        solver.solve_physics()
        X, Y, Az = solver.evaluate_grid((-6, 6), (-6, 6))
        
        vmin, vmax = np.percentile(Az, [1, 99])
        if abs(vmax - vmin) < 1e-9: vmax += 1e-9 
        
        fig_az = go.Figure(data=go.Contour(
            z=Az, x=X[0], y=Y[:,0], zmin=vmin, zmax=vmax, colorscale='Jet',
            hovertemplate="Re: %{x:.3f}<br>Im: %{y:.3f}<br>Az: %{z:.2f}<extra></extra>"
        ))
        
        for b in solver.boundaries:
            # 绘制黑圈物理圆周
            fig_az.add_trace(go.Scatter(
                x=np.real(b.c)+b.r*np.cos(np.linspace(0, 2*np.pi, 100)), 
                y=np.imag(b.c)+b.r*np.sin(np.linspace(0, 2*np.pi, 100)), 
                mode='lines', line=dict(color='black', width=2), hoverinfo='skip', showlegend=False
            ))
            # 物理条件标注
            fig_az.add_annotation(
                x=np.real(b.c), y=np.imag(b.c) + b.r, xref="x", yref="y",
                text=f"Γ{b.id}: I={b.u}A", showarrow=False, yanchor="bottom", yshift=1,
                font=dict(size=14, color="white"), borderpad=2
            )
            

        # 渲染 FIFO 队列高亮
        color_palette =['white', 'pink', 'yellow', 'lime', 'purple'] 
        for i, nid in enumerate(reversed(st.session_state.selected_nodes)):
            if i >= 5: break
            target_node = next((n for n in nodes if n.id == nid), None)
            if target_node:
                fig_az.add_trace(go.Scatter(
                    x=[np.real(target_node.z)], y=[np.imag(target_node.z)],
                    mode='markers+text',
                    marker=dict(color=color_palette[i], size=6, symbol='circle', line=dict(color='black', width=1.5)), 
                    text=[f"Gen{target_node.gen}-No.{target_node.idx}"],
                    textposition="top center", textfont=dict(color='green', size=12, weight='bold'),
                    hoverinfo='skip', showlegend=False
                ))
                
        fig_az.update_layout(title="矢量磁势 Az(x,y) 空间重构与追踪", width=600, height=500, yaxis_scaleanchor="x")
        st.plotly_chart(fig_az, use_container_width=True)
else:
    st.warning("👈 请点击左侧【播种并初始化】冷启动算法。")

    st.markdown("---")
    st.markdown("""
### 使用说明与系统操作指南

**1. 物理模型配置与冷启动**
在左侧控制面板设定各圆柱介质边界的内部总包围电流、几何参数以及介质反射算子 $K_m$。
（系统采用基于恒定磁场介质突变的绝对收敛模型）。
点击“播种并初始化”将清空现有缓存，并在设定的初始坐标点生成零阶场源。
*   **注：算法自适应播种**：系统仅会在电流 $I \\neq 0$ 的圆柱中心播种。若某边界设定为 0A，将作为纯感应边界参与迭代而不作为初始源。
若启用外部自由场源选项，还可在任意位置播种一个独立的线电流源，观察其对整体场分布的影响。

**2. 离散树状序列演化**
*   **单步繁衍**：单次执行广度优先搜索，触发像电流跨越边界的“映射裂变”（单节点严格分化为反演点与轴心点像源）。可用于观察误差残差的逐阶收缩过程。
*   **直接收敛**：引擎接管迭代循环，依据等比序列的达朗贝尔公比进行余项放缩评估。当级数截断误差上限低于容差阈值（默认限制单次最大繁衍10000节点以防内存溢出），算法自动阻断。

**3. 物理场与生成树的双向联动**
*   **场解析重构（左图）**：基于无矩阵求逆的正向叠加原理，渲染空间矢量磁势 $A_z$ 的等值线分布。粗黑实线为介质圆柱边界。
*   **关于矢量磁势标度（量纲归一化说明）**：
    真实矢量磁势 $A_z = -\\frac{\mu_{out} I}{2\pi} \ln r$。为了剥离未知环境变量 $\mu_{out}$ 的依赖，使数值具有普适意义，本计算系统已令真空磁导因子 $\\frac{\mu_{out}}{2\pi} \equiv 1$。图中z轴数值反映的为归一化后的相对磁势。
*   **定向高亮溯源**：在右侧 N 叉树拓扑图谱中点击任意节点，左侧物理空间域将高亮标注该像源的绝对物理坐标，并保留最近 5 次的 FIFO 历史查询队列。
""")
