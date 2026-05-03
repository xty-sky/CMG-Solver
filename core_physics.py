"""
core_physics.py
(已融合高精度防溢出与输入合法性阻断)
"""
import numpy as np

# 定义物理常数
EPSILON_0 = 8.8541878128e-12
MU_0 = 4 * np.pi * 1e-7

class SourceNode:
    def __init__(self, node_id: str, z: complex, q: float, parent_id: str = None, gen_by_boundary: int = None, generation: int = 0):
        self.node_id = node_id
        self.z = z
        self.q = q
        self.parent_id = parent_id
        self.gen_by_boundary = gen_by_boundary
        self.generation = generation  # 新增：记录 N叉树的层级深度

    def get_potential_at(self, z_eval: complex, physics_type: str = 'electrostatic') -> float:
        # 动态获取当前系统的浮点精度极限，防止极点爆炸
        eps_limit = np.finfo(float).eps
        distance = max(abs(z_eval - self.z), eps_limit)
        
        if physics_type == 'electrostatic':
            # \phi = -(q / 2\pi\epsilon) * \ln(r)
            return - (self.q / (2 * np.pi * EPSILON_0)) * np.log(distance)
        elif physics_type == 'magnetostatic':
            # A_z = -(I \mu / 2\pi) * \ln(r)
            return - (self.q * MU_0 / (2 * np.pi)) * np.log(distance)

class CircularBoundary:
    def __init__(self, b_id: int, center: complex, radius: float, potential: float, K_multiplier: float = -1.0):
        if radius <= 0:
            raise ValueError(f"数学错误：圆柱半径必须大于0，当前输入为 {radius}")
        self.id = b_id
        self.c = center
        self.r = radius
        self.u = potential
        self.K = K_multiplier

    def apply_reflection(self, source: SourceNode, new_node_id: str) -> SourceNode:
        z_old = source.z
        z_new = self.c + (self.r**2) / np.conj(z_old - self.c)
        q_new = source.q * self.K
        return SourceNode(new_node_id, z_new, q_new, source.node_id, self.id, source.generation + 1)

    def check_convergence_derivative(self, z_eval: complex) -> float:
        distance_sq = abs(z_eval - self.c)**2
        if distance_sq < np.finfo(float).eps:
            return float('inf') # 真实反映极点导数发散
        return (self.r**2) / distance_sq
class LinearBoundary:
    """
    直线/半平面边界 (退化情况：R -> infinity)
    保留用于支持平-球模型的扩展
    """
    def __init__(self, b_id: int, axis_real: float, potential: float, K_multiplier: float = -1.0):
        self.id = b_id
        self.x_axis = axis_real  # 假设为垂直于实轴的直线 x = axis_real
        self.u = potential
        self.K = K_multiplier

    def apply_reflection(self, source: SourceNode, new_node_id: str) -> SourceNode:
        """ 直线几何的共轭反演 Z_new = -conj(Z_old) + 2*x_axis """
        z_new = -np.conj(source.z) + 2 * self.x_axis
        q_new = source.q * self.K
        return SourceNode(new_node_id, z_new, q_new, source.node_id, self.id, source.generation + 1)
        
    def check_convergence_derivative(self, z_eval: complex) -> float:
        """ 直线反射的一阶导数模长始终为 1 (非严格压缩映射) """
        return 1.0