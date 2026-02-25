'''

import os
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from evaluator import evaluate_params_on_dataset

class GPTParameterOptimization(ElementwiseProblem):
    def __init__(self,):
        super().__init__(
            n_var = 5, #待优化参数
            n_obj = 3,#3个参数
            xl=np.array([0.0, 0.1, -2.0, -2.0, 50]),#下界
            xu=np.array([1.0, 1.0, 2.0, 2.0, 300]),#上界
            n_constr = 0#无约束
        )

    def _evaluate(self, x, out, *args, **kwargs):
        params = {
            'temp' : float(x[0]),
            'top_p' : float(x[1]),
            'freq_pen' : float(x[2]),
            'pres_pen' : float(x[3]),
            'max_tok' : int(round(x[4])),
        }
        f1, f2, f3 = evaluate_params_on_dataset(params)
        out["F"] = [f1, f2, f3]

class CheckpointCallback(Callback):
    def __init__(self, checkpoint_file="nsga2_checkpoint.npz",save_pareto_at=None):
        super().__init__()
        self.checkpoint_file = checkpoint_file
        self.save_pareto_at = set(save_pareto_at or [])
        self.gen = 0

    def notify(self, algorithm):
        np.savez(
            self.checkpoint_file,
            gen=algorithm.n_gen,  #当前已完成的代数
            pop_X=algorithm.pop.get("X"),
            pop_F=algorithm.pop.get("F"),
            archive_X=getattr(algorithm, 'archive_X', None),
        )
        print(f"检查点已保存，当前代际数：{algorithm.n_gen}")

        if self.gen in self.save_pareto_at:
            F = algorithm.pop.get("F")
            nds = NonDominatedSorting()
            fronts = nds.do(F)
            pareto_front_idx = fronts[0]

            X_pareto = algorithm.pop.get("X")[pareto_front_idx]
            F_pareto = F[pareto_front_idx]

            np.save(f"pareto_gen{algorithm.n_gen}_params.npy", X_pareto)
            np.save(f"pareto_gen{algorithm.n_gen}_objectives.npy", F_pareto)
            print(f"已保存第 {algorithm.n_gen} 代 Pareto 前沿，共 {len(X_pareto)} 个解")


def load_checkpoint(checkpoint_file="nsga2_checkpoint.npz"):
    if not os.path.exists(checkpoint_file):
        return None, 0

    data = np.load(checkpoint_file)
    gen = int(data['gen'])
    pop_X = data['pop_X']
    pop_F = data['pop_F']

    print(f"加载已存在检查点，已完成代际数：{gen}")
    return (pop_X, pop_F), gen

if __name__ == "__main__":
    SAVE_PARETO_AT = [3, 5, 7, 10]

    MAX_GEN = 10
    CHECKPOINT_FILE = "nsga2_checkpoint.npz"
    initial_pop, start_gen = load_checkpoint(CHECKPOINT_FILE)
    problem = GPTParameterOptimization()

    if start_gen == 0:
        sampling = FloatRandomSampling()
        print("Gen == 0开始寻优")
    else:
        sampling = initial_pop[0]
        MAX_GEN = MAX_GEN - start_gen
        print(f"Gen == {start_gen}开始寻优")

    algorithm = NSGA2(
        pop_size=40,
        sampling=sampling,
        crossover=SBX(prob=0.8, eta=10),
        mutation=PM(eta=20),
        eliminate_duplicates=True,  # 去除进化中完全相同的个体
    )

    callback = CheckpointCallback(checkpoint_file=CHECKPOINT_FILE)

    res = minimize(problem,
                   algorithm,
                   ("n_gen", MAX_GEN),#进化10代
                   callback = callback,
                   verbose=True,
                   seed=42,
                   )

    np.save("pareto_params.npy", res.X)
    np.save("pareto_objectives.npy", res.F)
    print(f"找到 {len(res.X)} 个 Pareto 最优解")
'''
import os
import numpy as np
import multiprocessing
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.population import Population

from evaluator import evaluate_params_on_dataset
class GPTParameterOptimization(ElementwiseProblem):
    def __init__(self, **kwargs):
        # 定义搜索空间：temp, top_p, freq_pen, pres_pen, max_tok
        xl = np.array([0.0, 0.1, -2.0, -2.0, 50])
        xu = np.array([1.0, 1.0, 2.0, 2.0, 300])
        super().__init__(n_var=5, n_obj=3, n_constr=0, xl=xl, xu=xu, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        # 将连续变量映射回 GPT 参数格式
        params = {
            'temp': float(x[0]),
            'top_p': float(x[1]),
            'freq_pen': float(x[2]),
            'pres_pen': float(x[3]),
            'max_tok': int(round(x[4])),
        }
        # 执行评估
        f1, f2, f3 = evaluate_params_on_dataset(params)
        out["F"] = [f1, f2, f3]


class CheckpointCallback(Callback):
    def __init__(self, checkpoint_file="nsga2_checkpoint.npz", save_pareto_at=None):
        super().__init__()
        self.checkpoint_file = checkpoint_file
        self.save_pareto_at = set(save_pareto_at or [])

    def notify(self, algorithm):
        gen = algorithm.n_gen

        # 保存检查点用于热启动
        np.savez(
            self.checkpoint_file,
            gen=gen,
            pop_X=algorithm.pop.get("X"),
            pop_F=algorithm.pop.get("F")
        )
        print(f">>> 检查点已保存：第 {gen} 代")

        # 定期保存 Pareto 前沿（非支配解）
        if gen in self.save_pareto_at:
            # 提取当前种群中的非支配解
            res = algorithm.opt
            X_pareto = res.get("X")
            F_pareto = res.get("F")

            np.save(f"pareto_gen{gen+7}_params.npy", X_pareto)
            np.save(f"pareto_gen{gen+7}_objectives.npy", F_pareto)
            print(f"[*] 已保存第 {gen} 代 Pareto 前沿，共 {len(X_pareto)} 个最优解")


def load_checkpoint(checkpoint_file):
    if not os.path.exists(checkpoint_file):
        return None, 0
    try:
        data = np.load(checkpoint_file)
        gen = int(data['gen'])
        pop_X = data['pop_X']
        pop_F = data['pop_F']
        # 封装为 pymoo 种群对象
        pop = Population.new("X", pop_X, "F", pop_F)
        return pop, gen
    except Exception as e:
        print(f"加载检查点失败: {e}")
        return None, 0

def run_optimization():
    # 配置参数
    SAVE_PARETO_AT = [3, 5, 7, 10]
    TOTAL_GENS = 10
    POP_SIZE = 40
    CHECKPOINT_FILE = "nsga2_checkpoint.npz"

    # 加载检查点
    initial_pop, start_gen = load_checkpoint(CHECKPOINT_FILE)
    problem = GPTParameterOptimization()

    if start_gen == 0:
        print("--- 启动全新优化任务 ---")
        algorithm = NSGA2(
            pop_size=POP_SIZE,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.8, eta=10),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
    else:
        print(f"--- 从第 {start_gen} 代恢复优化 ---")
        # 热启动：使用保存的种群初始化算法
        algorithm = NSGA2(
            pop_size=POP_SIZE,
            sampling=initial_pop,  # 直接传入 Population 对象
            crossover=SBX(prob=0.8, eta=10),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

    callback = CheckpointCallback(checkpoint_file=CHECKPOINT_FILE, save_pareto_at=SAVE_PARETO_AT)

    # 执行优化
    # 如果是从第 3 代恢复，总共 10 代，则还需要运行 7 代
    n_runs = TOTAL_GENS - start_gen
    if n_runs <= 0:
        print("任务已完成，无需继续运行。")
        return

    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', n_runs),
        callback=callback,
        verbose=True,
        seed=42
    )

    # 最终结果保存
    if res.X is not None:
        np.save("final_pareto_params.npy", res.X)
        np.save("final_pareto_objectives.npy", res.F)
        print(f"优化结束。共找到 {len(res.X)} 个 Pareto 最优解。")


if __name__ == "__main__":
    # 建议在评估逻辑复杂时开启多进程 (此处展示单进程，逻辑已解耦)
    run_optimization()