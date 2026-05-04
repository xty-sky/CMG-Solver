"""
field_solver.py
免矩阵求逆的纯顺向矢量磁势 (A_z) 网格渲染引擎
"""
import numpy as np

class FieldSolver:
    def __init__(self, boundaries, nodes):
        self.boundaries = boundaries
        self.nodes = nodes

    def evaluate_grid(self, x_range, y_range):
        """ 极简的解析场重构，直接暴露圆柱内部的磁位畸变 """
        X, Y = np.meshgrid(np.linspace(*x_range, 250), np.linspace(*y_range, 250))
        Z_grid = X + 1j * Y
        Az_grid = np.zeros_like(Z_grid, dtype=float)
        
        # 纯顺向累加：由于无须匹配边界恒定电位，直接利用真实的 q_topo (即物理电流) 叠加
        for n in self.nodes:
            # 引入 eps 防止 \ln(0) 引发底层 C 引擎 NaN 崩溃
            dist = np.maximum(np.abs(Z_grid - n.z), np.finfo(float).eps)
            # A_z \propto -I \ln(r)。(省略 \mu/2\pi 系数仅作归一化形态渲染)
            Az_grid += -n.q_topo * np.log(dist)
            
        # 柔性修剪极值：防止轴心奇点将颜色标尺(Colorbar)拉爆，但允许显示奇点漏斗
        vmin, vmax = np.percentile(Az_grid, [2, 98]) # 使用 2% 和 98% 分位数钳制
        margin = abs(vmax - vmin) * 0.2
        Az_grid = np.clip(Az_grid, vmin - margin, vmax + margin)
            
        return X, Y, Az_grid
