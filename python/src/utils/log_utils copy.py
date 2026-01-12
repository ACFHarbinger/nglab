"""
Logging utilities for training, evaluation, and simulation.

This module provides tools for:
- Configuring WandB logging.
- Creating and managing log directories.
- Logging metrics (scalar, histograms, heatmaps).
- Aggregating and visualizing log data.
"""

import os
import json
import wandb
import torch
import pickle
import datetime
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logic.src.utils.definitions as udef

from collections import Counter
from logic.src.utils.definitions import DAY_METRICS
from logic.src.utils.io_utils import read_json, compose_dirpath



def log_values(cost, grad_norms, epoch, batch_id, step, l_dict, tb_logger, opts):
    """
    Logs training metrics to the console, TensorBoard, and WandB.

    Args:
        cost (Tensor): The cost tensor.
        grad_norms (tuple): Tuple of gradient norms (all, clipped).
        epoch (int): Current epoch number.
        batch_id (int): Current batch identifier.
        step (int): Global step count.
        l_dict (dict): Dictionary of loss components.
        tb_logger: TensorBoard summary writer.
        opts (dict): Options dictionary.
    """
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print(
        "{}: {}, train_batch_id: {}, avg_cost: {}".format(
            "day" if opts["train_time"] else "epoch", epoch, batch_id, avg_cost
        )
    )
    print("grad_norm: {}, clipped: {}".format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts["no_tensorboard"]:
        tb_logger.log_value("avg_cost", avg_cost, step)
        tb_logger.log_value("actor_loss", l_dict["reinforce_loss"].mean().item(), step)
        tb_logger.log_value("nll", l_dict["nll"].mean().item(), step)
        tb_logger.log_value("grad_norm", grad_norms[0].item(), step)
        tb_logger.log_value("grad_norm_clipped", grad_norms_clipped[0], step)
        if "imitation_loss" in l_dict:
            tb_logger.log_value("imitation_loss", l_dict["imitation_loss"].item(), step)
        if opts["baseline"] == "critic":
            tb_logger.log_value("critic_loss", l_dict["baseline_loss"].item(), step)
            tb_logger.log_value("critic_grad_norm", grad_norms[1].item(), step)
            tb_logger.log_value(
                "critic_grad_norm_clipped", grad_norms_clipped[1].item(), step
            )

    if opts["wandb_mode"] != "disabled":
        wandb_data = {
            "avg_cost": avg_cost,
            "actor_loss": l_dict["reinforce_loss"].mean().item(),
            "nll": l_dict["nll"].mean().item(),
            "grad_norm": grad_norms[0].item(),
            "grad_norm_clipped": grad_norms_clipped[0],
        }
        if "imitation_loss" in l_dict:
            wandb_data["imitation_loss"] = l_dict["imitation_loss"].item()
        wandb.log(wandb_data)
        if opts["baseline"] == "critic":
            wandb.log(
                {
                    "critic_loss": l_dict["baseline_loss"].item(),
                    "critic_grad_norm": grad_norms[1].item(),
                    "critic_grad_norm_clipped": grad_norms_clipped[1].item(),
                }
            )

    if "imitation_loss" in l_dict and l_dict["imitation_loss"].item() != 0:
        print(f"imitation_loss: {l_dict['imitation_loss'].item():.6f}")

    return


def log_epoch(x_tup, loss_keys, epoch_loss, opts):
    """
    Logs summary statistics for a completed epoch.

    Args:
        x_tup (tuple): Tuple containing (label, value) for the x-axis (e.g., ("epoch", 1)).
        loss_keys (list): List of loss keys to log.
        epoch_loss (dict): Dictionary collecting losses for the epoch.
        opts (dict): Options dictionary.
    """
    log_str = f"Finished {x_tup[0]} {x_tup[1]} log:"
    for id, key in enumerate(loss_keys):
        if not epoch_loss.get(key):
            continue
        lname = key if key in udef.LOSS_KEYS else f"{key}_cost"
        # Handle cases where epoch_loss[key] might be empty or contain non-tensors
        try:
            lmean = torch.cat(epoch_loss[key]).float().mean().item()
        except Exception:
            lmean = 0.0

        log_str += f" {lname}: {lmean:.4f}"
        if opts["wandb_mode"] != "disabled":
            wandb.log({x_tup[0]: x_tup[1], lname: lmean}, commit=(key == loss_keys[-1]))
    print(log_str)
    return


def get_loss_stats(epoch_loss):
    """
    Computes mean, std, min, and max for each loss key in the epoch data.

    Args:
        epoch_loss (dict): Dictionary of loss sets.

    Returns:
        list: Flattened list of [mean, std, min, max] for each key.
    """
    loss_stats = []
    for key in epoch_loss.keys():
        loss_tensor = torch.cat(epoch_loss[key]).float()
        loss_tmp = [
            torch.mean(loss_tensor).item(),
            torch.std(loss_tensor).item(),
            torch.min(loss_tensor).item(),
            torch.max(loss_tensor).item(),
        ]
        loss_stats.extend(loss_tmp)
    return loss_stats


def log_training(loss_keys, table_df, opts):
    """
    Logs comprehensive training history to Parquet, WandB, and generates Plots.

    Args:
        loss_keys (list): List of loss column names.
        table_df (pd.DataFrame): DataFrame containing training stats.
        opts (dict): Options dictionary.
    """
    xname = "day" if opts["train_time"] else "epoch"
    x_values = [row_id for row_id in range(table_df.shape[0])]
    log_dir = os.path.join(
        opts["log_dir"],
        os.path.relpath(opts["save_dir"], start=opts["checkpoints_dir"]),
    )
    os.makedirs(log_dir, exist_ok=True)
    table_df.to_parquet(os.path.join(log_dir, "table.parquet"), engine="pyarrow")
    swapped_df = table_df.swaplevel(axis=1)
    swapped_df.columns = ["_".join(col).strip() for col in swapped_df.columns]
    if opts["wandb_mode"] != "disabled":
        wandb_table = wandb.Table(dataframe=swapped_df)
        wandb.log({"training_table": wandb_table})

    for l_id, l_key in enumerate(loss_keys):
        # mean_loss = np.array([row[wandb_table.columns.index(f"mean_{l_key}")] for row in wandb_table.data])
        # std_loss = np.array([row[wandb_table.columns.index(f"std_{l_key}")] for row in wandb_table.data])
        mean_loss = swapped_df[f"mean_{l_key}"].to_numpy()
        std_loss = swapped_df[f"std_{l_key}"].to_numpy()
        if np.all(mean_loss == std_loss):
            continue
        # max_loss = np.array([row[wandb_table.columns.index(f"max_{l_key}")] for row in wandb_table.data])
        # min_loss = np.array([row[wandb_table.columns.index(f"min_{l_key}")] for row in wandb_table.data])
        max_loss = swapped_df[f"max_{l_key}"].to_numpy()
        min_loss = swapped_df[f"min_{l_key}"].to_numpy()
        lower_bound = np.maximum(mean_loss - std_loss, min_loss)
        upper_bound = np.minimum(mean_loss + std_loss, max_loss)

        fig = plt.figure(
            l_id,
            figsize=(20, 10),
            facecolor="white",
            edgecolor="black",
            layout="constrained",
        )
        label = l_key if l_key in udef.LOSS_KEYS else f"{l_key}_cost"
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x_values, mean_loss, label=f"{label} μ", linewidth=2)
        ax.fill_between(
            x_values, lower_bound, upper_bound, alpha=0.3, label=f"{label} clip(μ ± σ)"
        )
        ax.scatter(x_values, mean_loss, color="red", s=50, zorder=5)
        ax.set_xlabel(xname)
        ax.set_ylabel(label)
        ax.set_title(f"{label} per {xname}")
        ax.legend()
        ax.grid(True, linestyle="-", alpha=0.9)
        fig_path = os.path.join(log_dir, f"{label}.png")
        fig.savefig(fig_path)
        plt.close(fig)
        if opts["wandb_mode"] != "disabled":
            wandb.log({label: wandb.Image(fig_path)})


def _sort_log(log):
    log = {key: value for key, value in sorted(log.items())}
    tmp_log = {}
    for key in log.keys():
        if "policy_last_minute" in key:
            tmp_log[key] = log[key]
    for key in log.keys():
        if "policy_regular" in key:
            tmp_log[key] = log[key]
    for key in log.keys():
        if "policy_look_ahead" in key:
            tmp_log[key] = log[key]
    for key in log.keys():
        if "gurobi" in key:
            tmp_log[key] = log[key]
    for key in log.keys():
        if "hexaly" in key:
            tmp_log[key] = log[key]
    for key in tmp_log.keys():
        log[key] = log.pop(key)
    return log


def sort_log(logfile_path, lock=None):
    """
    Sorts a JSON log file by grouping keys (policies, solvers).

    Args:
        logfile_path (str): Path to the log file.
        lock (threading.Lock, optional): Thread lock.
    """
    if lock is not None:
        lock.acquire(timeout=udef.LOCK_TIMEOUT)
    try:
        log = read_json(logfile_path, lock=None)
        log = _sort_log(log)
        with open(logfile_path, "w") as fp:
            json.dump(log, fp, indent=True)
    finally:
        if lock is not None:
            lock.release()
    return


def _convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.int64) or isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    return obj


def log_to_json(json_path, keys, dit, sort_log=True, sample_id=None, lock=None):
    """
    Writes data to a JSON file, handling updates and sorting.
    (Legacy version)

    Args:
        json_path (str): Path to the output JSON file.
        keys (list): List of keys for the inner dictionary.
        dit (dict): Data to write.
        sort_log (bool, optional): Whether to sort keys. Defaults to True.
        sample_id (int, optional): Index for list-based logs.
        lock (threading.Lock, optional): Thread lock.

    Returns:
        dict or list: The updated log data.
    """
    if lock is not None:
        lock.acquire(timeout=udef.LOCK_TIMEOUT)
    try:
        read_failed = False
        if os.path.isfile(json_path):
            # IMPORTANT: Pass lock=None to read_json to avoid double-locking,
            # since we've already acquired the lock here.
            try:
                old = read_json(json_path, lock=None)
            except json.JSONDecodeError:
                # Handle case where file is corrupt from a previous failed write.
                # If you can't read it, assume it's empty to prevent cascading errors.
                read_failed = True
                old = [] if "full" in json_path else {}

            if sample_id is not None and isinstance(old, list) and len(old) > sample_id:
                new = old[sample_id]
            elif isinstance(old, dict):
                new = old
            else:
                new = {}
        else:
            new = {}
            if sample_id is not None:
                old = []

        for key, val in dit.items():
            values = val.values() if isinstance(val, dict) else val
            new[key] = dict(zip(keys, values))

        if sort_log:
            new = _sort_log(new)
        if sample_id is not None:
            if isinstance(old, list):
                if len(old) > sample_id:
                    old[sample_id] = new
                else:
                    old.append(new)
            elif isinstance(old, dict):
                old[sample_id] = new
        else:
            old = new

        e = None
        write_error = None
        try:
            with open(json_path, "w") as fp:
                json.dump(_convert_numpy(old), fp, indent=True)
        except (TypeError, ValueError, FileNotFoundError) as e:
            write_error = e

        if read_failed or write_error is not None:
            # Handle error on write: Send output to a temporary file.
            timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
            filename, file_ext = os.path.splitext(json_path)
            tmp_path = filename + timestamp + "_TMP" + file_ext
            with open(tmp_path, "w") as fp_temp:
                json.dump(_convert_numpy(old), fp_temp, indent=True)

            if read_failed and write_error is not None:
                print(
                    f"\n[WARNING] Failed to read from and write to {json_path}.\n- Write Error: {e}"
                )
            elif read_failed:
                print(f"\n[WARNING] Failed to read from {json_path}.")
            else:
                assert write_error is not None
                print(
                    f"\n[WARNING] Failed to write to {json_path}.\n- Write Error: {e}"
                )

            print(f"Data saved to temporary file: {tmp_path}")
    finally:
        if lock is not None:
            lock.release()
    return old


def log_to_json2(json_path, keys, dit, sort_log=True, sample_id=None, lock=None):
    """
    Thread-safe and error-resilient JSON logger.
    Handles read/write errors gracefully with temporary file fallbacks.

    Args:
        json_path (str): Path to the output JSON file.
        keys (list): List of keys used if data items are sequences.
        dit (dict): Dictionary of data to log.
        sort_log (bool, optional): Whether to sort the log keys. Defaults to True.
        sample_id (int, optional): ID to update a specific list entry.
        lock (threading.Lock, optional): Thread lock.

    Returns:
        dict or list: The final data structure written (or attempted to write).
    """
    # 1. ACQUIRE LOCK (Protects the entire RMW cycle)
    if lock is not None:
        lock.acquire(timeout=udef.LOCK_TIMEOUT)

    data_to_write = None  # Variable to hold the final data structure to be written
    try:
        # --- READ AND MERGE DATA ---
        if os.path.isfile(json_path):
            try:
                # IMPORTANT: Pass lock=None as lock is already acquired here.
                old = read_json(json_path, lock=None)
            except json.JSONDecodeError:
                # ⚠️ 1. Handle JSONDecodeError on read: Log the ERROR, but ASSUME empty data
                # to continue processing the current thread's results.
                print(
                    f"\n[WARNING] JSONDecodeError reading: {json_path}. Assuming empty data for merge."
                )
                old = [] if "full" in json_path else {}

            if sample_id is not None and isinstance(old, list) and len(old) > sample_id:
                new = old[sample_id]
            elif isinstance(old, dict):
                new = old
            else:
                new = {}
        else:
            new = {}
            if sample_id is not None:
                old = []

        # --- MODIFY / MERGE NEW DATA ---
        for key, val in dit.items():
            values = val.values() if isinstance(val, dict) else val
            new[key] = dict(zip(keys, values))

        if sort_log:
            new = _sort_log(new)

        # Consolidate 'new' data back into 'old' structure
        if sample_id is not None:
            if isinstance(old, list):
                if len(old) > sample_id:
                    old[sample_id] = new
                else:
                    old.append(new)
            elif isinstance(old, dict):
                old[sample_id] = new
        else:
            old = new

        data_to_write = old  # Store the final merged data

        # --- WRITE DATA (With Fallback) ---
        try:
            with open(json_path, "w") as fp:
                json.dump(_convert_numpy(data_to_write), fp, indent=True)

        except (TypeError, ValueError) as e:
            # ⚠️ 2. Handle ANY error on write: Send output to a temporary file.
            timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
            temp_json_path = json_path + timestamp + "_TEMP.json"

            # Write to temporary file (still inside the lock for safety)
            with open(temp_json_path, "w") as fp_temp:
                json.dump(_convert_numpy(data_to_write), fp_temp, indent=True)

            print(
                f"\n[ERROR] Failed to write to {json_path}. Data saved to temporary file: {temp_json_path}"
            )
            print(f"Original Write Error: {e}")

    finally:
        # 3. RELEASE LOCK (Always released)
        if lock is not None:
            lock.release()

    return data_to_write


def log_to_pickle(pickle_path, log, lock=None, dw_func=None):
    """
    Logs data to a pickle file.

    Args:
        pickle_path (str): Output path.
        log (Any): Data to pickle.
        lock (threading.Lock, optional): Thread lock.
        dw_func (callable, optional): Post-write callback function.
    """
    if lock is not None:
        lock.acquire(timeout=udef.LOCK_TIMEOUT)
    try:
        file = open(pickle_path, "wb")
        pickle.dump(log, file)
        file.close()
        if dw_func is not None:
            dw_func(pickle_path)
    finally:
        if lock is not None:
            lock.release()
    return


def update_log(json_path, new_output, start_id, policies, sort_log=True, lock=None):
    """
    Updates specific entries in a JSON log file.

    Args:
        json_path (str): Path to the log file.
        new_output (list): New data to merge in.
        start_id (int): Starting index for the merge.
        policies (list): List of policies to update.
        sort_log (bool, optional): Whether to sort the result. Defaults to True.
        lock (threading.Lock, optional): Thread lock.

    Returns:
        list: The updated log data.
    """
    if lock is not None:
        lock.acquire(timeout=udef.LOCK_TIMEOUT)
    try:
        try:
            new_logs = read_json(json_path, lock=None)
        except json.JSONDecodeError:
            new_logs = [] if "full" in json_path else {}

        for id, log in enumerate(new_output):
            for pol in policies:
                new_logs[start_id + id][pol] = log[pol]

            if sort_log:
                new_logs[start_id + id] = _sort_log(new_logs[start_id + id])
        with open(json_path, "w") as fp:
            json.dump(new_logs, fp, indent=True)
    finally:
        if lock is not None:
            lock.release()
    return new_logs


@compose_dirpath
def load_log_dict(dir_paths, nsamples, show_incomplete=False, lock=None):
    """
    Loads log filenames for multiple graph sizes/runs.

    Args:
        dir_paths (list): List of directory paths.
        nsamples (list): Number of samples corresponding to each dir.
        show_incomplete (bool, optional): Check for incomplete runs. Defaults to False.
        lock (threading.Lock, optional): Thread lock.

    Returns:
        dict: Mapping of graph size to log file path.
    """
    assert len(dir_paths) == len(
        nsamples
    ), f"Len of dir_paths and nsamples lists must be equal, not {len(dir_paths)} != {len(nsamples)}"
    logs = {}
    for path, ns in zip(dir_paths, nsamples):
        gsize = int(os.path.basename(path).split("_")[1])
        logs[f"{gsize}"] = os.path.join(path, f"log_mean_{ns}N.json")
        if show_incomplete and ns > 1:
            counter = Counter()
            log_full = read_json(os.path.join(path, f"log_full_{ns}N.json"), lock)
            for run in log_full:
                counter.update(run.keys())

            for key, val in dict(counter).items():
                incomplete = False
                if ns - val > 0:
                    if not incomplete:
                        incomplete = not incomplete
                        print(f"graph {gsize} incomplete runs:")
                    print("-", key, "-", ns - val)
    return logs


def log_plot(visualize=False, **kwargs):
    """
    Execution function for saving static plots.

    Args:
        visualize (bool, optional): Whether to show the plot. Defaults to False.
        **kwargs: Must contain 'fig' (matplotlib Figure) and 'fig_filename' (str).
    """
    kwargs["fig"].savefig(kwargs["fig_filename"], bbox_inches="tight")
    if visualize:
        plt.show()
    plt.close(kwargs["fig"])
    return


@compose_dirpath
def output_stats(
    dir_path, nsamples, policies, keys, sort_log=True, print_output=False, lock=None
):
    """
    Calculates and saves mean and standard deviation of log data.

    Args:
        dir_path (str): Directory containing logs.
        nsamples (int): Number of samples.
        policies (list): Policies to analyze.
        keys (list): Keys to extract stats for.
        sort_log (bool, optional): Sort the output. Defaults to True.
        print_output (bool, optional): Print stats to console. Defaults to False.
        lock (threading.Lock, optional): Thread lock.

    Returns:
        tuple: (mean_dict, std_dict)
    """
    mean_filename = os.path.join(dir_path, f"log_mean_{nsamples}N.json")
    std_filename = os.path.join(dir_path, f"log_std_{nsamples}N.json")
    if lock is not None:
        lock.acquire(timeout=udef.LOCK_TIMEOUT)
    try:
        if os.path.isfile(mean_filename):
            mean_dit = read_json(mean_filename, lock=None)
            std_dit = read_json(std_filename, lock=None)
        else:
            mean_dit = {}
            std_dit = {}

        log = os.path.join(dir_path, f"log_full_{nsamples}N.json")
        data = read_json(log, lock=None)
        for pol in policies:
            tmp = []
            for n_id in range(nsamples):
                tmp.append(data[n_id][pol].values())
            mean_dit[pol] = {
                key: log for key, log in zip(keys, [*map(statistics.mean, zip(*tmp))])
            }
            std_dit[pol] = {
                key: log for key, log in zip(keys, [*map(statistics.stdev, zip(*tmp))])
            }

        if sort_log:
            mean_dit = _sort_log(mean_dit)
            std_dit = _sort_log(std_dit)
        if print_output:
            for lg, lg_std, pol in zip(
                mean_dit.values(), std_dit.values(), mean_dit.keys()
            ):
                logm = lg.values() if isinstance(lg, dict) else lg
                logs = lg_std.values() if isinstance(lg_std, dict) else lg_std
                tmp_lg = [(str(x), str(y)) for x, y in zip(logm, logs)]
                if pol in policies:
                    print(f"{pol}:")
                    for (x, y), key in zip(tmp_lg, keys):
                        print(
                            f"- {key} value: {x[:x.find('.')+3]} +- {y[:y.find('.')+5]}"
                        )

        with open(mean_filename, "w") as fp:
            json.dump(mean_dit, fp, indent=True)
        with open(std_filename, "w") as fp:
            json.dump(std_dit, fp, indent=True)
    finally:
        if lock is not None:
            lock.release()
    return mean_dit, std_dit


@compose_dirpath
def runs_per_policy(dir_paths, nsamples, policies, print_output=False, lock=None):
    """
    Counts successful runs for each policy in given directories.

    Args:
        dir_paths (list): List of directories.
        nsamples (list): List of expected sample counts.
        policies (list): List of policies.
        print_output (bool, optional): Print results. Defaults to False.
        lock (threading.Lock, optional): Thread lock.

    Returns:
        list: List of dictionaries of run IDs per policy.
    """
    assert len(dir_paths) == len(
        nsamples
    ), f"Len of dir_paths and nsamples lists must be equal, not {len(dir_paths)} != {len(nsamples)}"
    runs_ls = []
    for path, ns in zip(dir_paths, nsamples):
        dit = {pol: [] for pol in policies}
        log = os.path.join(path, f"log_full_{ns}N.json")
        data = read_json(log, lock)
        for id, run_data in enumerate(data):
            for key in dit:
                if key in run_data:
                    dit[key].append(id)

        runs_ls.append(dit)
        if print_output:
            gsize = int(os.path.basename(path).rsplit("_", 1)[1])
            print(f"graph {gsize} #runs per policy:")
            for key, val in dit.items():
                print(f"- {key}: {len(val)}")
                print(" -- sample IDs:", val)
    return runs_ls


def send_daily_output_to_gui(
    daily_log,
    policy,
    sample_idx,
    day,
    bins_c,
    collected,
    bins_c_after,
    log_path,
    tour,
    coordinates,
    lock=None,
):
    """
    Formats and writes daily simulation stats to a log file for the GUI.

    Args:
        daily_log (dict): Metrics for the day.
        policy (str): Policy name.
        sample_idx (int): Sample ID.
        day (int): Day index.
        bins_c (list): Bin levels before collection.
        collected (list): Amount collected.
        bins_c_after (list): Bin levels after collection.
        log_path (str): Output log file path.
        tour (list): Route indices.
        coordinates (pd.DataFrame or list): Node coordinates.
        lock (threading.Lock, optional): Thread lock.
    """
    full_payload = {k: v for k, v in daily_log.items() if k in DAY_METRICS[:-1]}
    route_coords = []

    # Prepare coordinates lookup: normalize headers and handle potential duplicates
    if isinstance(coordinates, pd.DataFrame):
        coords_lookup = coordinates.copy()
        # Normalize headers to UPPERCASE and STRIP whitespace
        coords_lookup.columns = [str(c).upper().strip() for c in coords_lookup.columns]
    else:
        # Fallback if coordinates is not a DataFrame (unlikely based on pipeline)
        coords_lookup = None

    for idx in tour:
        # Handle Depot (0) or Bins
        point_data = {"id": str(idx), "type": "bin"}

        if idx == 0:
            point_data["type"] = "depot"
            point_data["popup"] = "Depot"

        # Safe extraction of coordinates
        if coords_lookup is not None:
            # Check for ID in index (handling int vs str mismatch)
            lookup_idx = idx
            if lookup_idx not in coords_lookup.index:
                if str(idx) in coords_lookup.index:
                    lookup_idx = str(idx)
                elif (
                    isinstance(idx, (int, np.integer))
                    and int(idx) in coords_lookup.index
                ):
                    lookup_idx = int(idx)

            if lookup_idx in coords_lookup.index:
                try:
                    row = coords_lookup.loc[lookup_idx]
                    # If duplicate indices exist, take the first one
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]

                    # Look for standard Latitude/Longitude columns
                    lat_val = row.get("LAT", row.get("LATITUDE"))
                    lng_val = row.get("LNG", row.get("LONGITUDE", row.get("LONG")))

                    if lat_val is not None and lng_val is not None:
                        # Parse, replacing comma if locale issue
                        lat = float(str(lat_val).replace(",", "."))
                        lng = float(str(lng_val).replace(",", "."))
                        point_data["lat"] = lat
                        point_data["lng"] = lng

                        if idx != 0:
                            # Try to get a meaningful ID for popup, fallback to index
                            display_id = row.get("ID", idx)
                            point_data["popup"] = f"ID {display_id}"
                except Exception:
                    # Log error but continue so we don't crash the simulation
                    # print(f"Error parsing coordinates for node {idx}: {e}")
                    pass

        route_coords.append(point_data)

    full_payload.update({DAY_METRICS[-1]: route_coords})
    full_payload.update(
        {
            "bin_state_c": np.array(bins_c).tolist(),
            "bin_state_collected": np.array(collected).tolist(),
            "bins_state_c_after": np.array(bins_c_after).tolist(),
        }
    )
    log_msg = (
        f"GUI_DAY_LOG_START:{policy},{sample_idx},{day},{json.dumps(full_payload)}"
    )

    # Append the raw log message to a local file, immediately flushing the disk buffer.
    if lock is not None:
        lock.acquire(timeout=udef.LOCK_TIMEOUT)
    try:
        with open(log_path, "a") as f:
            f.write(log_msg + "\n")
            f.flush()
    except Exception as e:
        print(f"Warning: Failed to write to local log file: {e}")
    finally:
        if lock is not None:
            lock.release()


def send_final_output_to_gui(log, log_std, n_samples, policies, log_path, lock=None):
    """
    Formats and writes final simulation summary to a log file for the GUI.

    Args:
        log (dict): Mean stats.
        log_std (dict): Standard deviation stats.
        n_samples (int): Number of samples.
        policies (list): List of policies.
        log_path (str): Output log file path.
        lock (threading.Lock, optional): Thread lock.
    """
    lgsd = (
        {
            k: [[0] * len(v) if isinstance(v, tuple) else 0 for v in pol_data]
            for k, pol_data in log.items()
        }
        if log_std is None
        else {
            k: [list(v) if isinstance(v, tuple) else v for v in pol_data]
            for k, pol_data in log_std.items()
        }
    )
    summary_data = {
        "log": {
            k: [list(v) if isinstance(v, tuple) else v for v in pol_data]
            for k, pol_data in log.items()
        },
        "log_std": lgsd,
        "n_samples": n_samples,
        "policies": policies,
    }

    # 1. Standard GUI Output (for PySide capture)
    summary_message = f"GUI_SUMMARY_LOG_START: {json.dumps(summary_data)}"
    # print(summary_message)
    # sys.stdout.flush()

    # 2. Local Real-Time File Logging (Ensure final status is also written)
    if lock is not None:
        lock.acquire(timeout=udef.LOCK_TIMEOUT)
    try:
        with open(log_path, "a") as f:
            f.write(summary_message + "\n")
            f.flush()
    except Exception as e:
        print(f"Warning: Failed to write summary to local log file: {e}")
    finally:
        if lock is not None:
            lock.release()
