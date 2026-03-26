from __future__ import annotations

import sys
from pathlib import Path

from runner import run_selected_model


# ================= 用户在这里改参数 =================
# 可选：'point_mass' | 'nfw' | 'king' | 'p-jaffe' | 'none'
model_use = "none"

# 含 bestfit.dat 的目录（相对项目根目录或绝对路径）
source_dir = "work/SN_2Sersic_NFW"

# 统一输出目录（相对于 glade 目录）
results_root = "results"

# 通用覆盖参数：键名需与对应模型脚本中的顶层变量名一致
common_overrides = {
    # "active_subhalos": [1, 2, 3, 4],
    # "DE_MAXITER": 800,
    # "DE_POPSIZE": 75,
    # "MCMC_ENABLED": False,
}

# 模型专属参数覆盖
model_overrides = {
    "point_mass": {
        # "SEARCH_RADIUS": 0.075,
        # "MASS_LOG_RANGE": 0.8,
        # "OUTPUT_PREFIX": "v_pm_1_0",
    },
    "nfw": {
        # "SEARCH_RADIUS": 0.1,
        # "CONCENTRATION_GUESS": 8.0,
        # "OUTPUT_PREFIX": "v_nfw_2_0",
    },
    "king": {
        # "SEARCH_RADIUS": 0.1,
        # "MASS_LOG_GUESS": 6.0,
        # "OUTPUT_PREFIX": "v_king_1_0",
    },
    "p-jaffe": {
        # "SEARCH_RADIUS": 0.2,
        # "SIG_GUESS": 10.0,
        # "OUTPUT_PREFIX": "v_p_jaffe_2_0",
    },
    "none": {},
}


def main() -> int:
    glade_root = Path(__file__).resolve().parent
    try:
        return run_selected_model(
            model=model_use,
            glade_root=glade_root,
            source_dir=source_dir,
            common_cfg=common_overrides,
            model_cfg=model_overrides,
            results_root=results_root,
        )
    except Exception as exc:  # pragma: no cover - 作为主入口仅做错误提示
        print(f"[GLADE] 运行失败: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

