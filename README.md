# CMG-Solver
镜像法自检系统
# CMG-Solver: Generalized Image Method Semi-Analytical Solver

基于复解析映射与离散级数极限的拉普拉斯/泊松方程半数值求解系统 (Multi-Boundary Semi-Analytical Solver based on Asymptotic Limit of Discrete Image Series).

## 1. 物理框架与理论基座 (Theoretical Foundations)

本求解器不依赖有限元 (FEM) 或传统配置点拟合，其底层计算引擎严格基于偏微分方程 (PDE) 唯一性定理与极限分析学。核心理论支撑包括：

*   **Riccati 差分代数化归**：将三维平-球系统中的非线性坐标迭代 $z_n = -h + a^2/(h - z_{n-1})$ 转化为复交比等比数列，确立绝对收敛的显式解析解。
*   **Stolz-Cesàro 极限重构**：针对二维静电双圆柱系统，证明了等幅反演 ($\lambda_n = \lambda_{n-1}$) 必然导致的对数级数发散灾难。利用切萨罗平均提取调和特征基底 $D(Z) = \ln |(Z - Z_\beta)/(Z - Z_\alpha)|$，顺向重构二维物理场的封闭解析解。
*   **高斯平均值定理降维 (De-Collocation)**：彻底抛弃表面采样点拟合。通过边界圆周的复变积分 $\frac{1}{2\pi} \int_0^{2\pi} \Phi(Z_c + R e^{j\theta}) d\theta$，将泛函空间边界匹配降维为底层坐标的 $O(1)$ 代数距离判断。
*   **二次衰减算子与绝对收敛**：在稳磁场多介质模型中，证明同源像电流双重反演（闭合拓扑循环）产生的等比衰减公比为 $\rho = K_m^2 = (\frac{\mu_{in}-\mu_{out}}{\mu_{in}+\mu_{out}})^2 < 1$，利用达朗贝尔判别法界定系统的截断余项误差上界。

## 2. 核心架构与功能 (Architecture & Features)

*   **A-Priori Halting (先验硬熔断)**：依据调和函数极值定理，对退化边界条件（如 $U_i \equiv U_{const}$ 且域内无源）执行零算力拦截，直接输出平凡解。
*   **N-ary Tree BFS Engine (N叉树广度优先生成)**：利用巴拿赫压缩映射原理，实时计算雅可比范数 $|f'(Z)|$，监控莫比乌斯映射算子的收敛性并生成离散像源拓扑。
*   **Dual-View Interaction (双域交互联动)**：
    *   *物理空间域*：利用 Numpy 与 Plotly 渲染高精度等势线（动态修剪奇点防止极值爆炸）。
    *   *拓扑逻辑域*：渲染纯数学代数生成的树状网络。
    *   *FIFO 溯源*：捕获拓扑树节点点击事件，在物理空间逆向投影并高亮最近 5 阶代数演化轨迹。

## 3. 部署与运行 (Installation & Usage)

环境要求：`Python 3.8+`

**安装依赖**：
```bash
pip install numpy streamlit plotly networkx
```
启动系统：
在项目根目录终端执行以下指令，浏览器将自动唤醒交互界面：
```bash
streamlit run app.py
```
## 4. 目录结构 (Structure)
采用单文件高内聚架构，便于学术交流与复现：
app.py: 包含底层算子 SourceNode/CircularBoundary、迭代引擎 ImageTreeEngine、场解算器 FieldSolver 以及 Streamlit 前端交互视图。
