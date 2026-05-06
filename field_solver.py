"""
field_solver.py
"""
import numpy as np

class FieldSolver:
    def __init__(self, boundaries, nodes):
        self.boundaries = boundaries
        self.nodes = nodes

    def evaluate_grid(self, x_range, y_range):
        X, Y = np.meshgrid(np.linspace(*x_range, 250), np.linspace(*y_range, 250))
        Z_grid = X + 1j * Y
        Az_grid = np.zeros_like(Z_grid, dtype=float)
        
        for n in self.nodes:
            # 引入 eps 防止 \ln(0) 引发底层 C 引擎 NaN 崩溃
            dist = np.maximum(np.abs(Z_grid - n.z), np.finfo(float).eps)
            # A_z \propto -I \ln(r)。(省略 \mu/2\pi 系数仅作归一化形态渲染)
            Az_grid += -n.q_topo * np.log(dist)
            # 但允许显示奇点漏斗
        vmin, vmax = np.percentile(Az_grid, [2, 98]) 
        margin = abs(vmax - vmin) * 0.2
        Az_grid = np.clip(Az_grid, vmin - margin, vmax + margin)
            
        return X, Y, Az_grid
