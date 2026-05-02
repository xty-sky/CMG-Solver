"""
tree_engine.py
多边界离散镜像序列 N 叉树 (N-ary Tree) 生成引擎与敛散自检控制器
"""
import numpy as np
from core_physics import SourceNode, CircularBoundary, LinearBoundary

class TrivialSolutionException(Exception):
    """ 触发极值定理退化条件时抛出的自定义异常 """
    pass

class ImageTreeEngine:
    def __init__(self, boundaries: list, tolerance: float = 1e-5):
        """
        :param boundaries: 物理边界对象集合 \partial\Omega
        :param tolerance: 工程容差阈值 \epsilon_{tol}
        """
        self.boundaries = boundaries
        self.tolerance = tolerance
        
        # 存储拓扑层级状态空间: list of lists
        # self.layers[k] 对应理论中的第 k 阶状态向量 S_k
        self.layers =[] 
        self.current_layer_idx = -1
        
        self.node_counter = 0  # 追踪生成的节点总数

    def check_a_priori_conditions(self):
        """
        先验判定 1：调和函数极值原理 (Maximum Principle)
        检测所有狄利克雷边界电位是否一致
        """
        if not self.boundaries:
            return
            
        u_vals = [b.u for b in self.boundaries]
        # 判断边界极差是否趋于 0
        if max(u_vals) - min(u_vals) < 1e-12:
            raise TrivialSolutionException(
                f"检测到平凡解边界: 域内全空间电势恒为 {u_vals[0]}, 停止算力分配。"
            )

    def auto_seed_roots(self):
        """
        零阶势场初始播种 (Seeding)
        依据理论 3.1.1 节证明：初始场源必须严格定位于圆心（几何解析中心），以保证等势正交性。
        """
        initial_layer =[]
        for b in self.boundaries:
            if isinstance(b, CircularBoundary):
                # 初始等效强度暂赋归一化值 1.0，真实物理量由场域重构器联立求解
                root_node = SourceNode(
                    node_id=f"root_{b.id}",
                    z=b.c,
                    q=1.0, 
                    gen_by_boundary=b.id
                )
                initial_layer.append(root_node)
                self.node_counter += 1
                
        self.layers.append(initial_layer)
        self.current_layer_idx = 0

    def step_forward(self) -> bool:
        """
        执行一次广度优先搜索 (BFS)，生成 S_{k+1} = \cup T_i(S_k \setminus S_from_i)
        返回 True 表示继续；返回 False 表示已收敛，停止繁衍。
        """
        if self.current_layer_idx < 0:
            raise RuntimeError("引擎尚未播种，无法繁衍！请先调用 auto_seed_roots()")

        current_nodes = self.layers[self.current_layer_idx]
        next_layer =[]
        
        for node in current_nodes:
            for b in self.boundaries:
                # 避免对原始边界连续反演的回路熔断机制
                if node.gen_by_boundary == b.id:
                    continue
                    
                # 提取复导数模长 L(Z) 用于巴拿赫收缩判定
                l_z = b.check_convergence_derivative(node.z)
                if l_z >= 1.0 and isinstance(b, CircularBoundary):
                    # 若 L(Z) >= 1，此时发生散度爆炸，系统必须切入 Stolz-Cesaro 正则化模式
                    # 但在底层节点生成引擎中，仍可生成坐标，由外部渲染器执行截断操作
                    pass 

                new_id = f"node_{self.node_counter}"
                new_node = b.apply_reflection(node, new_id)
                next_layer.append(new_node)
                self.node_counter += 1

        self.layers.append(next_layer)
        self.current_layer_idx += 1
        
        # 误差自检判定
        return not self.evaluate_convergence()

    def evaluate_convergence(self) -> bool:
        """
        达朗贝尔比值放缩与级数余项自检 (D'Alembert's Ratio & Remainder Bound)
        依据理论 4.5 节的截断放缩判定。
        """
        if self.current_layer_idx < 1:
            return False  # 层数太浅，继续生成
            
        current_layer = self.layers[self.current_layer_idx]
        prev_layer = self.layers[self.current_layer_idx - 1]
        
        if not current_layer or not prev_layer:
            return True

        # 计算当前层相比于上一层的平均电荷绝对值衰减公比 \rho
        mean_q_curr = np.mean([abs(n.q) for n in current_layer])
        mean_q_prev = np.mean([abs(n.q) for n in prev_layer])
        
        rho = mean_q_curr / mean_q_prev if mean_q_prev > 0 else 0
        
        # 若 rho >= 1，系统处于静电等幅反演状态 (发散域)
        # 此时基于余项的误差评估失效，必须由后续场求解器提取切萨罗极限定值，故本引擎仅按最大深度强制截断
        if rho >= 0.999:
            return False 

        # 在收敛域 (如恒定磁场 \rho < 1) 下，执行严格的几何级数余项放缩 E(N) <= M * \rho^(N+1) / (1-\rho)
        max_q_curr = np.max([abs(n.q) for n in current_layer])
        error_bound = max_q_curr * rho / (1.0 - rho)
        
        if error_bound <= self.tolerance:
            print(f"-> 自检触发: 第 {self.current_layer_idx} 层截断余项上界 {error_bound:.2e} <= 容差 {self.tolerance:.2e}")
            return True # 误差达标，通知主循环停止繁衍
            
        return False

    def get_all_nodes(self) -> list:
        """ 铺平多维数组，返回用于网格渲染的所有有效场源节点 """
        flattened = [node for layer in self.layers for node in layer]
        return flattened