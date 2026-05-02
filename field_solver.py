"""
field_solver.py
多边界半解析镜像场的代数求解器与网格渲染引擎
"""
import numpy as np
import matplotlib.pyplot as plt

class FieldSolver:
    def __init__(self, boundaries: list, image_nodes: list):
        """
        :param boundaries: 物理边界对象列表
        :param image_nodes: 由 TreeEngine 生成的包含全部离散像源的节点列表
        """
        self.boundaries = boundaries
        self.nodes = image_nodes
        
        # 将节点列表转化为字典，利用拓扑树回溯每个像源的“根种子(Root Seed)”
        self.node_dict = {n.node_id: n for n in self.nodes}
        self._assign_roots()

    def _assign_roots(self):
        """ 图论回溯：向上遍历 N 叉树，判定每个节点属于哪个根节点发出的基底 """
        for n in self.nodes:
            curr = n
            while curr.parent_id is not None:
                curr = self.node_dict[curr.parent_id]
            n.root_id = curr.node_id  # 绑定基底归属

    def solve_true_seeds(self):
        """
        依据偏微分方程唯一性定理，通过边界配置法（Collocation Method）求解真实的种子系数。
        构建并解线性方程组: [A] *[Lambda] = [U]
        """
        if not self.nodes:
            return  # 若无节点直接跳过，防止空矩阵报错
        
        m = len(self.boundaries)
        A = np.zeros((m, m))
        U = np.zeros(m)
        
        # 1. 提取各个独立边界作为测试点 (Test Points)
        # 为避开奇点，在每个边界圆周上取一点（例如右侧极点）
        test_points =[b.c + b.r * np.exp(1j * 0) for b in self.boundaries]
        
        # 获取所有唯一的根节点 ID
        root_ids = list(set([n.root_id for n in self.nodes]))
        root_ids.sort() # 保证矩阵列的对应顺序一致
        
        # 2. 组装传递矩阵 A: A_{ij} 表示第 j 个根节点序列在第 i 个测试点产生的基底电势
        for i, z_test in enumerate(test_points):
            U[i] = self.boundaries[i].u  # 目标边界条件
            
            for j, root_id in enumerate(root_ids):
                # 提取属于第 j 个基底的所有节点
                basis_nodes = [n for n in self.nodes if n.root_id == root_id]
                phi_ij = 0.0
                for node in basis_nodes:
                    # 调取底层物理核心的对数势计算公式
                    phi_ij += node.get_potential_at(z_test)
                A[i, j] = phi_ij
                
        # 3. 代数求解 [Lambda] = A^{-1} * [U]
        try:
            true_lambdas = np.linalg.solve(A, U)
        except np.linalg.LinAlgError:
            raise RuntimeError("传递矩阵奇异！检测到不可解的退化边界，请重新设置拓扑。")
            
        # 4. 乘数回代：将解得的真实物理系数强制赋给所有像源
        for j, root_id in enumerate(root_ids):
            lambda_true = true_lambdas[j]
            for node in self.nodes:
                if node.root_id == root_id:
                    node.q *= lambda_true  # 纯顺向代数映射：基底振幅放缩
                    
        print(f"-> 代数方程组求解完成，真实种子系数已锚定: {true_lambdas}")

    def evaluate_grid(self, x_range: tuple, y_range: tuple, resolution: int = 400):
        """
        构建离散空间网格，并执行泛函重构
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z_grid = X + 1j * Y
        Phi_grid = np.zeros_like(Z_grid, dtype=float)
        
        # 叠加所有确定系数的像源格林函数
        for node in self.nodes:
            dist = np.abs(Z_grid - node.z)
            dist[dist < 1e-12] = 1e-12 # 防止奇点溢出
            Phi_grid += -node.q * np.log(dist)
            
        # 奇点掩膜 (Masking)：利用逻辑数组，剔除圆柱边界内部的非物理求解域
        mask = np.zeros_like(Phi_grid, dtype=bool)
        for b in self.boundaries:
            # 找到在边界内部的网格点
            inside = np.abs(Z_grid - b.c) <= b.r
            mask = mask | inside
            # 将边界内部的电势强制钳制为该边界的恒定电势 (符合理想导体物理现实)
            Phi_grid[inside] = b.u 
            
        return X, Y, Phi_grid

    def render_field(self, X, Y, Phi_grid):
        """ 学术级 Matplotlib 图表渲染引擎 """
        plt.figure(figsize=(8, 6), dpi=120)
        
        # 1. 绘制等势线高保真填充 (Contourf)
        # levels = np.linspace(np.min(Phi_grid), np.max(Phi_grid), 40) lead to fierce color gradation
        vmin, vmax = min([b.u for b in self.boundaries]), max([b.u for b in self.boundaries])
        levels = np.linspace(vmin, vmax, 40)
        contour = plt.contourf(X, Y, Phi_grid, levels=levels, cmap='jet', alpha=0.85)
        plt.colorbar(contour, label='Potential $\Phi$')
        
        # 2. 绘制边界物理轮廓
        theta = np.linspace(0, 2*np.pi, 100)
        for b in self.boundaries:
            plt.plot(np.real(b.c) + b.r*np.cos(theta), 
                     np.imag(b.c) + b.r*np.sin(theta), 'k-', linewidth=2)
            plt.text(np.real(b.c), np.imag(b.c), f'U={b.u}', 
                     color='white', fontweight='bold', ha='center', va='center')
            
        # 3. 渲染所有离散映射像点 (节点溯源)
        z_coords = [n.z for n in self.nodes]
        plt.scatter(np.real(z_coords), np.imag(z_coords), c='white', s=10, 
                    edgecolors='black', marker='x', label='Image Nodes')
        
        plt.title("Semi-Analytical Potential Field by Generalized Image Method", fontweight='bold')
        plt.xlabel("Real Axis (x)")
        plt.ylabel("Imaginary Axis (y)")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    #输出 Plotly 的函数     
    def get_plotly_contour(self, X, Y, Phi_grid):
        """ 生成 Web 原生的 Plotly 交互式等势线图 """
        import plotly.graph_objects as go
        
        # 吸收审查建议与 Bug 修补：采用边界极值锁定 Colorbar，防颜色跳变
        vmin, vmax = min([b.u for b in self.boundaries]), max([b.u for b in self.boundaries])
        # 【微观加固】：防止等电位导致 vmin==vmax 引起绘图引擎除零崩溃
        if abs(vmax - vmin) < 1e-5:
            vmax += 1e-5
            
        fig = go.Figure(data=go.Contour(
            z=Phi_grid, x=X[0], y=Y[:,0],
            colorscale='Jet',
            zmin=vmin, zmax=vmax, # 强制钳制极值
            contours=dict(
                coloring='heatmap',
                showlines=True
            )
        ))
        
        # 绘制边界圆
        theta = np.linspace(0, 2*np.pi, 100)
        for b in self.boundaries:
            fig.add_trace(go.Scatter(
                x=np.real(b.c) + b.r*np.cos(theta),
                y=np.imag(b.c) + b.r*np.sin(theta),
                mode='lines', line=dict(color='black', width=3),
                hoverinfo='none', name=f'Boundary {b.id}'
            ))
            
        fig.update_layout(
            title="Semi-Analytical Potential Field",
            xaxis_title="Re(Z)",
            yaxis_title="Im(Z)",
            width=700, height=600,
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        return fig
# ================= 测试运行用例 =================
if __name__ == "__main__":
    from core_physics import CircularBoundary
    from tree_engine import ImageTreeEngine
    
    # 构建物理边界
    b1 = CircularBoundary(b_id=1, center=-3+0j, radius=1.5, potential=100, K_multiplier=0.95)
    b2 = CircularBoundary(b_id=2, center=3+0j, radius=1.0, potential=-50, K_multiplier=0.95)
    
    # 启动迭代引擎 (软熔断)
    engine = ImageTreeEngine([b1, b2], tolerance=1e-3)
    engine.auto_seed_roots()
    for _ in range(8):  # 迭代深度
        if not engine.step_forward():
            break
            
    # 移交求解器
    solver = FieldSolver([b1, b2], engine.get_all_nodes())
    solver.solve_true_seeds() # 计算确切物理系数
    
    # 空间网格离散与渲染
    X, Y, Phi = solver.evaluate_grid(x_range=(-6, 6), y_range=(-4, 4))
    solver.render_field(X, Y, Phi)
    # 若在终端单独测试此文件，仍可调用静态绘图查看
    import matplotlib.pyplot as plt
    plt.contourf(X, Y, Phi, levels=40, cmap='jet')
    plt.colorbar()
    plt.show()