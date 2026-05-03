"""
app.py
多边界广义镜像阵列法 半数值解析求解系统 
包含：图层拦截防穿透、点击联动高亮、严格序号标记
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ================= 1. 底层物理与拓扑引擎 =================

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

    def reflect(self, source: SourceNode, new_id: str, new_idx: int) -> SourceNode:
        z_new = self.c + (self.r**2) / np.conj(source.z - self.c)
        q_new = source.q_topo * self.K
        return SourceNode(new_id, z_new, q_new, source.id, self.id, source.gen + 1, source.root_id, new_idx)

class ImageTreeEngine:
    def __init__(self, boundaries, tolerance=1e-3):
        self.boundaries = boundaries
        self.tol = tolerance
        self.layers =[]
        self.node_cnt = 0

    def seed_roots(self):
        layer_0 =[]
        idx = 1
        for b in self.boundaries:
            root_id = f"root_{b.id}"
            node = SourceNode(root_id, b.c, 1.0, gen_by_boundary=b.id, root_id=root_id, idx_in_gen=idx)
            layer_0.append(node)
            self.node_cnt += 1
            idx += 1
        self.layers.append(layer_0)

    def step_forward(self):
        curr_nodes = self.layers[-1]
        next_layer =[]
        idx = 1
        for node in curr_nodes:
            for b in self.boundaries:
                if node.gen_by == b.id: continue 
                next_layer.append(b.reflect(node, f"n_{self.node_cnt}", idx))
                self.node_cnt += 1
                idx += 1
        self.layers.append(next_layer)
        return self._check_convergence()

    def _check_convergence(self):
        if len(self.layers) < 2: return False
        curr_q = np.mean([abs(n.q_topo) for n in self.layers[-1]])
        prev_q = np.mean([abs(n.q_topo) for n in self.layers[-2]])
        rho = curr_q / prev_q if prev_q > 0 else 0
        
        if rho >= 0.999: return False 
        
        max_q = np.max([abs(n.q_topo) for n in self.layers[-1]])
        error_bound = max_q * rho / (1.0 - rho)
        return error_bound <= self.tol

    def get_nodes(self):
        return[n for layer in self.layers for n in layer]

# ================= 2. 场域解算与渲染器 =================

class FieldSolver:
    def __init__(self, boundaries, nodes):
        self.boundaries = boundaries
        self.nodes = nodes
        self.lambdas = {} 

    def solve_physics(self):
        """
        依据调和函数高斯平均值定理 (Gauss's Mean Value Theorem)，
        直接通过底层坐标运算建立代数方程组，彻底规避配置点采样拟合。
        """
        if not self.nodes: return
        
        m = len(self.boundaries)
        A, U = np.zeros((m, m)), np.zeros(m)
        root_ids = sorted(list(set(n.root_id for n in self.nodes)))
        
        # 遍历每一个目标物理边界 (组装矩阵行)
        for i, b in enumerate(self.boundaries):
            U[i] = b.u  # 目标常数电势
            
            # 遍历每一组种子基底 (组装矩阵列)
            for j, r_id in enumerate(root_ids):
                basis_nodes =[n for n in self.nodes if n.root_id == r_id]
                phi_ij = 0.0
                
                # 抛弃采样点，利用解析积分恒等式直接计算边界电势
                for node in basis_nodes:
                    dist_to_center = abs(node.z - b.c)
                    
                    if dist_to_center <= b.r:
                        # 像源在圆内 (或正好在边界上)，积分解析解取决于圆半径
                        eff_dist = b.r
                    else:
                        # 像源在圆外，积分解析解取决于到圆心的距离
                        eff_dist = dist_to_center
                        
                    # 纯代数累加，不传入任何空间变量 Z
                    phi_ij += -node.q_topo * np.log(eff_dist)
                    
                A[i, j] = phi_ij
                
        # 矩阵求逆解算真实物理系数
        try:
            lambdas = np.linalg.solve(A, U)
            self.lambdas = {r_id: lambdas[j] for j, r_id in enumerate(root_ids)}
        except np.linalg.LinAlgError:
            self.lambdas = {r_id: 1.0 for r_id in root_ids} # 退化情况：直接赋予默认系数，后续渲染会反映出边界条件不满足的异常状态 

    def evaluate_grid(self, x_range, y_range):
        X, Y = np.meshgrid(np.linspace(*x_range, 250), np.linspace(*y_range, 250))
        Z_grid = X + 1j * Y
        Phi = np.zeros_like(Z_grid, dtype=float)
        
        for n in self.nodes:
            phys_q = n.q_topo * self.lambdas[n.root_id]
            # 引入 eps 极限，防止 \ln(0) 引发底层 C 引擎 NaN 崩溃
            dist = np.maximum(np.abs(Z_grid - n.z), np.finfo(float).eps)
            Phi += -phys_q * np.log(dist)
            
        # 暴露圆柱内部的离散对数奇点场！
        # 仅通过 Numpy 的 clip 柔性修剪极值，防止等势线颜色标尺(Colorbar)被无穷大拉爆
        vmin, vmax = min([b.u for b in self.boundaries]), max([b.u for b in self.boundaries])
        # 动态宽容度，允许显示比边界高50%的内部奇点峰值，同时设置绝对下限防止过度压缩
        margin = max(abs(vmax - vmin) * 0.5, 30.0) 
        Phi = np.clip(Phi, vmin - margin, vmax + margin)
            
        return X, Y, Phi

def plot_tree(nodes):
    """ 树状图渲染，注入 customdata 以备跨组件联动提取 """
    layers = {}
    for n in nodes:
        layers.setdefault(n.gen,[]).append(n)
        
    pos, x_coords, y_coords, texts, colors, custom_ids = {}, [], [], [], [],[]
    for gen, gen_nodes in layers.items():
        
        y_vals = np.linspace(1, -1, len(gen_nodes)+2)[1:-1] if len(gen_nodes)>1 else [0]
        for i, n in enumerate(gen_nodes):
            pos[n.id] = (gen, y_vals[i])
            x_coords.append(gen)
            y_coords.append(y_vals[i])
            colors.append(gen)
            custom_ids.append(n.id) # 埋入节点唯一ID
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
        customdata=custom_ids, # 关键注入点：用于交互捕获
        hovertext=texts, hoverinfo='text',
        marker=dict(size=12, color=colors, colorscale='Viridis', showscale=True, colorbar=dict(title='Generation'))
    ))
    fig.update_layout(title="N-ary Tree 繁衍图谱 (点击节点在左图亮显)", showlegend=False, 
                      xaxis=dict(title="迭代代数 (Generation)", tick0=0, dtick=1), 
                      yaxis=dict(showticklabels=False), margin=dict(l=10, r=10, t=40, b=10),
                      dragmode='pan') 
    return fig

# ================= 3. Streamlit 交互页面 =================
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
    st.header("⚙️ 动态边界配置舱")
    
    # 动态渲染参数表单，彻底去耦合
    for i, cfg in enumerate(st.session_state.b_configs):
        st.markdown(f"**边界 {cfg['id']}**")
        c1, c2, c3, c4 = st.columns(4)
        cfg['u'] = c1.number_input("U (V)", value=cfg['u'], step=10.0, key=f"u_{i}")
        cfg['r'] = c2.number_input("半径 R", value=cfg['r'], min_value=0.1, step=0.1, key=f"r_{i}")
        cfg['x'] = c3.number_input("圆心 x", value=cfg['x'], step=0.5, key=f"x_{i}")
        cfg['y'] = c4.number_input("圆心 y", value=cfg['y'], step=0.5, key=f"y_{i}")
        st.markdown("---")
        
    k_m = st.slider("反射算子 (Km)", 0.1, 1.0, 0.80, 0.01)
    
    if st.button("🔄 播种并初始化 (Reset)", use_container_width=True):
        st.session_state.boundaries =[]
        for cfg in st.session_state.b_configs:
            b = CircularBoundary(cfg['id'], complex(cfg['x'], cfg['y']), cfg['r'], cfg['u'], k_m)
            st.session_state.boundaries.append(b)
            
        st.session_state.engine = ImageTreeEngine(st.session_state.boundaries, tolerance=1e-4)
        st.session_state.engine.seed_roots()
        st.session_state.converged = False

    col1, col2 = st.columns(2)
    if col1.button("▶️ 单步繁衍"):
        if st.session_state.engine:
            st.session_state.converged = st.session_state.engine.step_forward()
    if col2.button("⏭️ 直接收敛"):
        if st.session_state.engine:
            # 放宽最大保护上限，适应多边界深层迭代
            while not st.session_state.engine.step_forward() and st.session_state.engine.node_cnt < 2000: 
                pass
            st.session_state.converged = True
    # [新增] 清除功能键：一键清空高亮追踪 FIFO 队列
    if st.button("🧹 清除高亮轨迹 (Clear)", use_container_width=True):
        st.session_state.selected_nodes =[]        
if st.session_state.engine:
    nodes = st.session_state.engine.get_nodes()
    
    # 【逆向渲染流设计】：为捕获右侧点击事件，必须在代码中先渲染右图，再渲染左图
    c1, c2 = st.columns([1.2, 1])
    
    # --- 渲染右侧拓扑树，并捕获点击事件 ---
    with c2:
        fig_tree = plot_tree(nodes)
        # on_select="rerun" 捕获鼠标点击，触发脚本重载
        # [修正] 增加 selection_mode="points"，在 Web 通信层彻底封杀框选与套索事件
        selection_event = st.plotly_chart(fig_tree, use_container_width=True, on_select="rerun", selection_mode="points", key="tree_click")
                
        selected_node_id = None
        if selection_event and hasattr(selection_event.selection, "points"):
            points = selection_event.selection.points
            if len(points) > 0:
                pt = points[0]
                # 兼容字典格式与对象格式，防止 AttributeError
                selected_node_id = pt.get("customdata") if isinstance(pt, dict) else getattr(pt, "customdata", None)
    
                # [新增] FIFO 队列状态更新逻辑
                if selected_node_id:
                    # 如果点了旧节点，先移除再加到队尾（刷新为最新）
                    if selected_node_id in st.session_state.selected_nodes:
                        st.session_state.selected_nodes.remove(selected_node_id)
                    st.session_state.selected_nodes.append(selected_node_id)
                    # 强制维持最大长度为 5
                    if len(st.session_state.selected_nodes) > 5:
                        st.session_state.selected_nodes.pop(0)
    
    # --- 渲染左侧物理场，并根据点击事件执行高亮 ---
    with c1:
        solver = FieldSolver(st.session_state.engine.boundaries, nodes)
        solver.solve_physics()
        X, Y, Phi = solver.evaluate_grid((-6, 6), (-6, 6))
        
        vmin, vmax = min(b.u for b in solver.boundaries), max(b.u for b in solver.boundaries)
        if abs(vmax - vmin) < 1e-9: vmax += 1e-9 
            
        fig_phi = go.Figure(data=go.Contour(
            z=Phi, x=X[0], y=Y[:,0], zmin=vmin, zmax=vmax, colorscale='Jet',
            hovertemplate="Re: %{x:.3f}<br>Im: %{y:.3f}<br>Φ: %{z:.2f}V<extra></extra>" # 定制底层悬停信息
        ))
        
        for b in solver.boundaries:
            fig_phi.add_trace(go.Scatter(
                x=np.real(b.c)+b.r*np.cos(np.linspace(0, 2*np.pi, 100)), 
                y=np.imag(b.c)+b.r*np.sin(np.linspace(0, 2*np.pi, 100)), 
                mode='lines', line=dict(color='black', width=2), 
                hoverinfo='skip', showlegend=False # 【BUG 修复】：强制穿透，防止 trace 0 拦截
            ))
            # [修正]：坐标沿虚轴平移 +R，并利用 yanchor 将文本框底部贴合圆周正上方
            fig_phi.add_annotation(
                x=np.real(b.c), 
                y=np.imag(b.c) + b.r, # 数学定位：圆心 Y 坐标加上半径 R
                xref="x", yref="y",
                text=f"Γ{b.id}: U={b.u}V", # 纯文本：去加粗、去换行
                showarrow=False,
                yanchor="bottom", # 渲染定位：强制文本框的下边缘锚定在目标坐标上
                yshift=1,         # 视觉微调：向上悬浮 3 像素，防止字底贴线
                font=dict(size=14, color="white"),
                # bgcolor="rgba(0, 0, 0, 0.5)",  # 仅保留半透明遮罩防隐身，去除所有边框
                borderpad=2
            )        # 散点隐式渲染 (不干扰视觉)
        z_coords = [n.z for n in nodes]
        fig_phi.add_trace(go.Scatter(
            x=np.real(z_coords), y=np.imag(z_coords), mode='markers', 
            marker=dict(color='white', symbol='x', size=4, opacity=0.3), 
            hoverinfo='skip', showlegend=False
        ))

# 【核心修正】：FIFO 多彩历史追踪渲染 (最多显示5点，尺寸缩小并带黑框)
        color_palette = ['white', 'pink', 'yellow', 'lime', 'purple'] # 索引0代表最新点击
        
        # 逆序遍历 FIFO，使得最新点击的点使用调色板靠前的颜色
        for i, nid in enumerate(reversed(st.session_state.selected_nodes)):
            if i >= 5: break
            target_node = next((n for n in nodes if n.id == nid), None)
            if target_node:
                fig_phi.add_trace(go.Scatter(
                    x=[np.real(target_node.z)], y=[np.imag(target_node.z)],
                    mode='markers+text',
                    marker=dict(
                        color=color_palette[i], 
                        size=6,           # 缩小尺寸
                        symbol='circle', 
                        line=dict(color='black', width=1.5) # 明显的黑色实线圆框
                    ), 
                    # 防拥挤：仅为最新的前两个点显示文字标签——暂时失能
                    text=[f"Gen{target_node.gen}-No.{target_node.idx}"]# if i < 2 else[]
                    ,
                    textposition="top center",
                    textfont=dict(color='green', size=12, weight='bold'),
                    hoverinfo='skip', showlegend=False
                ))
                
        fig_phi.update_layout(title="物理空间域：双向交互式场解析重构", width=600, height=500, yaxis_scaleanchor="x")
        st.plotly_chart(fig_phi, use_container_width=True)
else:
    st.warning("👈 请点击左侧【播种并初始化】冷启动算法。")
# ================= 6. 底部预留说明区 =================
st.markdown("---")
st.markdown("""
### 📌 使用说明
1. 在左侧配置舱设定各个边界的电位与位置参数，并调整反射衰减算子。
2. 点击 **[播种并初始化]** 按钮，系统将自动于解析几何圆心生成初始场源种子。
3. 点击 **[单步繁衍]** 追踪 N 叉树拓扑生长过程，或点击 **[直接收敛]** 获取迭代最大次数的电场分布状态。
4. **交互联动**：在右侧 N 叉树繁衍图谱中**点击任意节点**，左侧物理空间域将高亮显示其历史轨迹（最多5个最近点击节点，再点击新的节点，最先点击的一个将消失）。

### 📊 图像解释
* **物理空间域**：展示多连通域边界系统内的电势分布等高线。粗黑实线框为真实设定的狄利克雷物理边界。
* **拓扑逻辑域**：将离散像源生成过程剥离物理坐标，降维映射为纯拓扑代数生成树（N-ary Tree）。

### 🔄 更新日志
* **v1.2**：扩展物理视图 Y 轴界限；加入 FIFO 五节点历史彩色追踪系统；新增说明基座。
* **v1.1**：模块化解耦侧边栏，支持动态边界参数调优；修复底层字典解析抛出错误。
* **v1.0**：泛函拓扑树与双视图交互引擎全链路贯通。
""")
