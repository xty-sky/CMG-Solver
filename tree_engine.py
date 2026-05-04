"""
tree_engine.py
支持映射裂变与双模态播种的 N 叉树生成引擎
"""
import numpy as np
from core_physics import SourceNode, CircularBoundary

class ImageTreeEngine:
    def __init__(self, boundaries, tolerance=1e-3):
        self.boundaries = boundaries
        self.tol = tolerance
        self.layers =[]
        self.node_cnt = 0

    def seed_roots(self, external_sources=None):
        """ 双模态场源播种 (Seeding) """
        layer_0 =[]
        idx = 1
        
        # 模态 II：各个边界包围的已知电流，在几何轴心自动播种
        for b in self.boundaries:
            if b.I != 0: # 若包围电流为0则无须播种
                root_id = f"root_B{b.id}"
                # 直接将真实的物理电流 b.I 作为种子强度
                node = SourceNode(root_id, b.c, b.I, gen_by_boundary=b.id, root_id=root_id, idx_in_gen=idx)
                layer_0.append(node)
                self.node_cnt += 1
                idx += 1
                
        # 模态 I：可选的外部自由线电流场源
        if external_sources:
            for i, ext in enumerate(external_sources):
                root_id = f"root_Ext{i}"
                node = SourceNode(root_id, ext['z'], ext['I'], gen_by_boundary=-1, root_id=root_id, idx_in_gen=idx)
                layer_0.append(node)
                self.node_cnt += 1
                idx += 1
                
        self.layers.append(layer_0)

    def step_forward(self):
        if not self.layers: return True
        curr_nodes = self.layers[-1]
        next_layer =[]
        idx = 1
        
        for node in curr_nodes:
            for b in self.boundaries:
                if node.gen_by == b.id: continue 
                
                # 映射裂变：一个父节点生成两个子节点 (T 和 C)
                fission_nodes = b.reflect(node, f"n_{self.node_cnt}", idx)
                next_layer.extend(fission_nodes)
                
                self.node_cnt += 2
                idx += 2
                
        if not next_layer: return True # 无新节点生成
        self.layers.append(next_layer)
        return self._check_convergence()

    def _check_convergence(self):
        if len(self.layers) < 2: return False
        
        # 计算每一层电流的绝对值衰减公比
        curr_q = np.mean([abs(n.q_topo) for n in self.layers[-1]])
        prev_q = np.mean([abs(n.q_topo) for n in self.layers[-2]])
        rho = curr_q / prev_q if prev_q > 0 else 0
        
        if rho >= 0.999: return False # 发散域不作等比余项判断
        
        max_q = np.max([abs(n.q_topo) for n in self.layers[-1]])
        error_bound = max_q * rho / (1.0 - rho)
        return error_bound <= self.tol

    def get_nodes(self):
        return[n for layer in self.layers for n in layer]
