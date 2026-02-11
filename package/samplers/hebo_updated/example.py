import json
import optuna
import plotly.graph_objects as go

from sampler import HEBOSampler

search_space = {
    "x": optuna.distributions.FloatDistribution(-10, 10),
    "y": optuna.distributions.IntDistribution(-10, 10),
    "z": optuna.distributions.FloatDistribution(-5, 5),
    "w": optuna.distributions.CategoricalDistribution(["a", "b", "c", "d", "e", "f"]),
    "flag": optuna.distributions.CategoricalDistribution([True, False]),
    "log_p": optuna.distributions.FloatDistribution(1e-3, 1e1, log=True),
    "theta": optuna.distributions.FloatDistribution(0.05, 1.0, step=0.05),
    "mode": optuna.distributions.CategoricalDistribution(["fast", "balanced", "slow"]),
    "budget": optuna.distributions.IntDistribution(1, 5),
}


def objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", -10, 10)
    z = trial.suggest_float("z", -5, 5)
    w = trial.suggest_categorical("w", ["a", "b", "c", "d", "e", "f"])
    flag = trial.suggest_categorical("flag", [True, False])
    log_p = trial.suggest_float("log_p", 1e-3, 1e1, log=True)
    theta = trial.suggest_float("theta", 0.05, 1.0, step=0.05)
    mode = trial.suggest_categorical("mode", ["fast", "balanced", "slow"])
    budget = trial.suggest_int("budget", 1, 5)

    base = (
        x**2
        + y**2
        + 0.5 * z**2
        + 2 * log_p
        + (1 / theta)
        + 0.2 * budget
    )
    categorical_penalty = {"a": 0, "b": 1, "c": 2}.get(w, 3)
    mode_penalty = {"fast": 2, "balanced": 1, "slow": 0}[mode]
    return base + categorical_penalty + mode_penalty + (0 if flag else 0.5)


if __name__ == "__main__":
    sampler = HEBOSampler(search_space=search_space)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=25)
    print(study.best_trial.params)

    fig = optuna.visualization.plot_optimization_history(study)
    history = sampler.get_surrogate_prediction_history()
    if not history.empty:
        trial_ids = history.get("_trial_number")
        if trial_ids is None:
            trial_ids = list(range(len(history)))
        fig.add_trace(
            go.Scatter(
                x=trial_ids,
                y=history["_hebo_pred_mean"],
                mode="markers+lines",
                marker=dict(size=6, color="royalblue", line=dict(width=1)),
                line=dict(color="royalblue", dash="dash"),
                name="HEBO pred mean",
                error_y=dict(
                    type="data",
                    array=history["_hebo_pred_std"],
                    visible=True,
                    thickness=1.5,
                    width=3,
                    color="lightseagreen",
                ),
            )
        )
    fig.write_html("hebo_optimization_history.html")
    history = sampler.get_surrogate_prediction_history()
    if not history.empty:
        spaces_column = history.get("_search_space")
        unique_spaces = (
            spaces_column.dropna().unique() if spaces_column is not None else []
        )
        if unique_spaces.size:
            print("\nSearch spaces passed to the HEBO surrogate:")
            for idx, serialized in enumerate(unique_spaces):
                space = json.loads(serialized)
                print(f"  space {idx}:")
                print(json.dumps(space, indent=2))
        print("\nHEBO surrogate prediction history (real scale parameters + mean/std):")
        for i, row in history.iterrows():
            params = {
                k: row[k]
                for k in row.index
                if not (
                    k.startswith("_hebo_pred_")
                    or k in {"_trial_number", "_search_space"}
                )
            }
            mean = row["_hebo_pred_mean"]
            std = row["_hebo_pred_std"]
            trial_label = (
                f"(trial {int(row['_trial_number'])})"
                if "_trial_number" in row.index
                else ""
            )
            print(
                f" Candidate {i} {trial_label}: params={params}, mean={mean:.6f}, std={std:.6f}"
            )
