import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

params = np.load("data/fig/pareto_gen10_params.npy")
objectives = np.load("data/fig/pareto_gen10_objectives.npy")

for i, (p, obj) in enumerate(zip(params, objectives)):
    print(f"解 {i+1}:")
    print(f" 参数: temp={p[0]:.2f}, top_p={p[1]:.2f}, freq_pen={p[2]:.2f}, pres_pen={p[3]:.2f}, max_tok={int(p[4])}")
    print(f" 句法F1={1-obj[0]:.3f}, 压缩率={obj[1]:.3f}, 关系F1={1-obj[2]:.3f}\n")

syntax_f1 = 1-objectives[:, 0]
compress_ratio = objectives[:, 1]
re_f1 = 1-objectives[:, 2]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    syntax_f1,
    compress_ratio,
    re_f1,
    c=syntax_f1 + re_f1 - compress_ratio,
    cmap='viridis',
    s=60,
    alpha=0.8
)

ax.set_xlabel('Syntax F1', fontsize=12)
ax.set_ylabel('Compression Ratio', fontsize=12)
ax.set_zlabel('Relation Extraction F1', fontsize=12)

plt.title('Pareto Front: Multi-objective Optimization of GPT Parameters(gen-10)', fontsize=14)
cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
cbar.set_label('Composite Score (F1↑, Ratio↓)', fontsize=10)
ax.view_init(elev=20, azim=45)
ax.grid(True, linestyle='--', alpha=0.5)
plt.savefig('pareto_front_gen10_3d.png', dpi=300, bbox_inches='tight')