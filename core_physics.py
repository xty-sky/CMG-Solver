"""
core_physics.py
多边界稳磁场同构与映射裂变底层物理算子
"""
import numpy as np

class SourceNode:
    def __init__(self, node_id, z, q_topo, parent_id=None, gen_by_boundary=None, generation=0, root_id=None, idx_in_gen=1):
        self.id = node_id
        self.z = z
        self.q_topo = q_topo  # 在当前算法下，它直接代表真实的物理线电流 I
        self.parent_id = parent_id
        self.gen_by = gen_by_boundary
        self.gen = generation
        self.root_id = root_id
        self.idx = idx_in_gen

class CircularBoundary:
    def __init__(self, b_id, center, radius, current_I, K_mult):
        self.id = b_id
        self.c = center
        self.r = radius
        self.I = current_I  # 物理边界条件：圆柱内包含的总真实电流 I
        self.K = K_mult

    def reflect(self, source: SourceNode, base_new_id: str, start_idx: int) -> list:
        """
        【核心突破：映射裂变 (Mapping Fission)】
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
