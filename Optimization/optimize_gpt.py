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
        xl = np.array([0.0, 0.1, -2.0, -2.0, 50])
        xu = np.array([1.0, 1.0, 2.0, 2.0, 300])
        super().__init__(n_var=5, n_obj=3, n_constr=0, xl=xl, xu=xu, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        params = {
            'temp': float(x[0]),
            'top_p': float(x[1]),
            'freq_pen': float(x[2]),
            'pres_pen': float(x[3]),
            'max_tok': int(round(x[4])),
        }

        f1, f2, f3 = evaluate_params_on_dataset(params)
        out["F"] = [f1, f2, f3]


class CheckpointCallback(Callback):
    def __init__(self, checkpoint_file="nsga2_checkpoint.npz", save_pareto_at=None):
        super().__init__()
        self.checkpoint_file = checkpoint_file
        self.save_pareto_at = set(save_pareto_at or [])

    def notify(self, algorithm):
        gen = algorithm.n_gen

        np.savez(
            self.checkpoint_file,
            gen=gen,
            pop_X=algorithm.pop.get("X"),
            pop_F=algorithm.pop.get("F")
        )
        print(f">>> 检查点已保存：第 {gen} 代")

        if gen in self.save_pareto_at:
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
        pop = Population.new("X", pop_X, "F", pop_F)
        return pop, gen
    except Exception as e:
        print(f"加载检查点失败: {e}")
        return None, 0

def run_optimization():
    SAVE_PARETO_AT = [3, 5, 7, 10]
    TOTAL_GENS = 10
    POP_SIZE = 40
    CHECKPOINT_FILE = "nsga2_checkpoint.npz"

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
        algorithm = NSGA2(
            pop_size=POP_SIZE,
            sampling=initial_pop,
            crossover=SBX(prob=0.8, eta=10),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

    callback = CheckpointCallback(checkpoint_file=CHECKPOINT_FILE, save_pareto_at=SAVE_PARETO_AT)
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

    if res.X is not None:
        np.save("final_pareto_params.npy", res.X)
        np.save("final_pareto_objectives.npy", res.F)
        print(f"优化结束。共找到 {len(res.X)} 个 Pareto 最优解。")


if __name__ == "__main__":
    run_optimization()