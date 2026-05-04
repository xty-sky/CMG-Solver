"""
main_gui.py
基于 Tkinter 的广义镜像阵列半数值解析求解器交互界面
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core_physics import CircularBoundary
from tree_engine import ImageTreeEngine, TrivialSolutionException
from field_solver import FieldSolver

class CMGSolverGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("广义镜像法半数值计算系统 (N-ary Tree Solver)")
        self.geometry("1100x750")
        
        # 物理与引擎引用
        self.boundaries =[]
        self.engine = None
        
        self._setup_ui()
        self._init_physics_system()

    def _setup_ui(self):
        """ 划分左右区域：左侧为控制面板 (Controller)，右侧为渲染视图 (View) """
        # --- 左侧控制面板 ---
        self.ctrl_frame = tk.Frame(self, width=280, bg="#f0f0f0", relief=tk.SUNKEN, borderwidth=2)
        self.ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.ctrl_frame.pack_propagate(False)
        
        tk.Label(self.ctrl_frame, text="求解器控制面板", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=15)
        
        self.btn_reset = tk.Button(self.ctrl_frame, text="初始化系统 (Reset Seeding)", 
                                   command=self._init_physics_system, font=("Arial", 11), bg="#d0e8f1")
        self.btn_reset.pack(fill=tk.X, padx=20, pady=10)
        
        self.btn_step = tk.Button(self.ctrl_frame, text="单步迭代繁衍 (Step Forward)", 
                                  command=self._step_iterate, font=("Arial", 11, "bold"), bg="#d4f1d0")
        self.btn_step.pack(fill=tk.X, padx=20, pady=10)
        
        # 状态回显标签
        self.status_var = tk.StringVar()
        self.status_label = tk.Label(self.ctrl_frame, textvariable=self.status_var, 
                                     font=("Consolas", 10), justify=tk.LEFT, bg="#f0f0f0", anchor="w")
        self.status_label.pack(fill=tk.BOTH, padx=20, pady=20)
        
        # --- 右侧渲染视图 ---
        self.view_frame = tk.Frame(self, bg="white")
        self.view_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.view_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _init_physics_system(self):
        """ 初始化边界几何参数与 N 叉树引擎 """
        self.ax.clear()
        
        # 定义真实物理边界 (依据模型二、三设定)
        # K_multiplier=0.95 模拟介质突变带来的能量损耗 (绝对收敛域)
        b1 = CircularBoundary(b_id=1, center=-3+0j, radius=1.5, potential=100, K_multiplier=0.95)
        b2 = CircularBoundary(b_id=2, center=3+0j, radius=1.0, potential=-50, K_multiplier=0.95)
        self.boundaries = [b1, b2]
        
        self.engine = ImageTreeEngine(self.boundaries, tolerance=1e-3)
        
        try:
            self.engine.check_a_priori_conditions()
            self.engine.auto_seed_roots()
            self._update_display(first_init=True)
        except TrivialSolutionException as e:
            messagebox.showinfo("退化熔断", str(e))
            
    def _step_iterate(self):
        """ 执行广度优先搜索生成一层节点，并触发场域重算 """
        if self.engine is None:
            return
            
        try:
            is_converged = not self.engine.step_forward()
            self._update_display()
            
            if is_converged:
                self.btn_step.config(state=tk.DISABLED)
                messagebox.showinfo("自检通知", "达朗贝尔级数余项已低于工程容差，算法完成逼近！")
        except Exception as e:
            messagebox.showerror("运行错误", str(e))

    def _update_display(self, first_init=False):
        """ 接驳 FieldSolver 求解真实系数并利用 Matplotlib 更新视图 """
        nodes = self.engine.get_all_nodes()
        
        # 1. 组装并求解线性代数方程组 (获取真实物理极化系数)
        solver = FieldSolver(self.boundaries, nodes)
        solver.solve_true_seeds()
        
        # 2. 网格化并剥离奇点
        X, Y, Phi = solver.evaluate_grid(x_range=(-6, 6), y_range=(-4, 4), resolution=250)
        
        # 3. 渲染操作
        self.ax.clear()
        # 绘制等势线填充
        contour = self.ax.contourf(X, Y, Phi, levels=45, cmap='jet', alpha=0.85)
        if first_init and not hasattr(self, 'cbar'):
            self.cbar = self.fig.colorbar(contour, ax=self.ax, label='Potential / V')
        else:
            self.cbar.update_normal(contour)
            
        # 绘制狄利克雷物理边界
        theta = np.linspace(0, 2*np.pi, 100)
        for b in self.boundaries:
            self.ax.plot(np.real(b.c) + b.r*np.cos(theta), 
                         np.imag(b.c) + b.r*np.sin(theta), 'k-', linewidth=2)
            self.ax.text(np.real(b.c), np.imag(b.c), f'U={b.u}', 
                         color='white', fontweight='bold', ha='center', va='center')
            
        # 绘制离散镜像阵列点
        z_coords = [n.z for n in nodes]
        self.ax.scatter(np.real(z_coords), np.imag(z_coords), c='white', s=15, 
                        edgecolors='black', marker='x')
        
        self.ax.set_title(f"Iteration Depth: {self.engine.current_layer_idx} | Total Nodes: {len(nodes)}", fontweight='bold')
        self.ax.set_xlabel("Re(Z)")
        self.ax.set_ylabel("Im(Z)")
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.axis('equal')
        
        self.canvas.draw()
        
        # 更新状态回显面板
        status_text = (
            f"=== 算法状态 ===\n"
            f"迭代深度: {self.engine.current_layer_idx}\n"
            f"场源节点数: {len(nodes)}\n"
            f"单步矩阵阶数: {len(self.boundaries)}x{len(self.boundaries)}\n\n"
            f"=== 数学特征 ===\n"
            f"级数公比: K_m = {self.boundaries[0].K}\n"
            f"容差阈值: {self.engine.tolerance}\n"
            f"状态: {'已收敛' if self.btn_step['state'] == tk.DISABLED else '逼近中...'}"
        )
        self.status_var.set(status_text)

if __name__ == "__main__":
    app = CMGSolverGUI()
    app.mainloop()
