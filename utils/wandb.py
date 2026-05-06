"""Downloads all logged runs from WandB and stores them into a dataframe."""

from __future__ import annotations

import fire
import pandas as pd
import wandb
from tqdm import tqdm

WANDB_PROJECT = "implicit_vi"
WANDB_ENTITY = "implicit-vi"


def download_results(
    file: str = "results/experiment_results.csv",
    group: str | None = None,
    created_after: str | None = None,
    keys: list[str] | None = None,
):
    """Download results from Weights and Biases.

    :param file: File name.
    :param group: WandB group of the experiment.
    :param created_after: Only download runs created after the specified date, e.g.'2024-01-01T##'.
    :param keys: Which columns of the run table to return.
    """
    api = wandb.Api(timeout=100)

    # Project is specified by <entity/project-name>
    filters = {}
    if group is not None:
        filters["group"] = group
    if created_after is not None:
        filters["$and"] = [
            {"created_at": {"$lt": "2099-01-01T##", "$gt": created_after}}
        ]
    runs = api.runs(
        WANDB_ENTITY + "/" + WANDB_PROJECT,
        filters=filters,
        # per_page=1000,
    )

    all_runs_df_list = []

    for run in tqdm(runs):

        # Logged values for all steps from this run
        run_df = pd.DataFrame(
            run.scan_history(keys=keys),
            # page_size=100,
        )
        # run_df = run.history()

        run_df.insert(0, "run name", run.name)
        run_df.insert(1, "experiment", run.group)
        run_df.insert(
            2, "gpu", None if run.metadata is None else run.metadata.get("gpu", None)
        )
        try:
            runtime_s = run.summary["_runtime"]
        except KeyError:
            # If the run is not finished, we cannot get the runtime.
            runtime_s = None
        run_df.insert(3, "Runtime (s)", runtime_s)

        # run.config is the input metrics.
        # We remove special values that start with _.
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        for idx, (k, v) in enumerate(config.items()):
            if isinstance(v, list):
                v = str(v)
            run_df.insert(idx + 3, k, v)

        all_runs_df_list.append(run_df)

    all_runs_df = pd.concat(all_runs_df_list)
    all_runs_df.to_csv(file)


if __name__ == "__main__":
    fire.Fire(download_results)
