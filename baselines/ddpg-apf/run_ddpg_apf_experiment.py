import argparse
import csv
import datetime
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from BaseClass.CalMod import XML2Dict
from FactoryClass.EnvFactory import EnvFactory


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_simulator_config():
    return XML2Dict("./config/PathPlan_City.xml")["simulator"]


def make_env(simulator_config=None, *, map_level=None):
    simulator_config = load_simulator_config() if simulator_config is None else simulator_config
    config = simulator_config["env"]
    if map_level is not None:
        config = dict(config, manuscript_map_level=map_level)
    return EnvFactory().Create_Env(config)


def train_step_if_ready(uav):
    if len(uav.Trainer.replay_memory.buffer) <= uav.Trainer.Batch_Size:
        return None
    b_s, b_a, b_r, b_ns, b_d, idx, weights = uav.Trainer.replay_memory.sample2(uav.Trainer.Batch_Size)
    uav.transition_dict = {
        "states": b_s,
        "actions": b_a,
        "next_states": b_ns,
        "rewards": b_r,
        "dones": b_d,
        "idx": idx,
        "weights": weights,
    }
    return uav.Train_nn()


def run_episode(env, mode, seed, eps=0.1, train=False, render_html=False, *, map_episode=None):
    set_seed(seed)
    scene = None
    if map_episode is None:
        env.Scene_Random_Reset()
    else:
        scene = env.shared_scene_reset(map_episode)
    uav = env.Agents[0]
    uav.Trainer.Is_Train = 1 if train and mode == "ddpg" else 0
    baseline_action = uav.ddpg_apf_coefficients_to_action(
        uav._float_param("k_att_baseline", 3.5),
        uav._float_param("k_rep_baseline", 3.0),
    )

    losses = []
    done = False
    info = "normal"
    while not done:
        state = uav.state()
        if mode == "baseline":
            action = baseline_action
        else:
            action = uav.get_action(state, eps)

        next_state, reward, done, info = env.Move_Agent(0, action)
        if train and mode == "ddpg":
            uav.Trainer.replay_memory.add(state, action, reward, next_state, done)
            result = train_step_if_ready(uav)
            if result is not None:
                loss = result["loss"]
                losses.append(float(loss.detach().cpu().item() if hasattr(loss, "detach") else loss))

    html_path = ""
    if render_html:
        render_data = {"uuid": f"ddpg_apf_{mode}_{seed}"}
        env.render(render_data)
        html_path = render_data.get("html_dir", "")

    row = {
        "mode": mode,
        "seed": seed,
        "html_path": html_path,
        "map_len": env.len,
        "map_width": env.width,
        "map_height": env.h,
        "obstacle_count": len(env.buildings),
        "fixed_obstacle_height": getattr(env, "fixed_obstacle_height", "") if scene is None else "",
        "info": info,
        "success": 1 if info == "success" else 0,
        "path_len": uav.path_len,
        "steps": uav.Step,
        "score": uav.total_score,
        "loss": losses[-1] if losses else "",
        "k_att": getattr(uav, "last_k_att", ""),
        "k_rep": getattr(uav, "last_k_rep", ""),
    }
    if scene is not None:
        row.update(map_level=scene.level, map_episode=scene.episode, map_seed=scene.seed)
    return row


def summarize(rows, mode):
    selected = [row for row in rows if row["mode"] == mode]
    if not selected:
        return {"success_rate": 0.0, "path_len": 0.0, "score": 0.0}
    return {
        "success_rate": sum(row["success"] for row in selected) / len(selected),
        "path_len": sum(row["path_len"] for row in selected) / len(selected),
        "score": sum(row["score"] for row in selected) / len(selected),
    }


def log_train_row(writer, row, step):
    loss = row["loss"] if row["loss"] != "" else 0
    writer.add_scalar("Train/Average_Score", row["score"], step)
    writer.add_scalar("Train/Loss", loss, step)
    writer.add_scalar("Train/Success_Rate", row["success"], step)
    writer.add_scalar("Train/Collision", 1 if row["info"] == "meet_threaten" else 0, step)


def main():
    parser = argparse.ArgumentParser(description="Train and compare paper-style DDPG-APF against fixed-parameter APF.")
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--map-level", type=int, choices=(2, 3),
                        help="Use shared manuscript scenes for new baseline runs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--no-render-html", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    simulator_config = load_simulator_config()
    if args.train_episodes is None:
        args.train_episodes = int(simulator_config.get("num_episodes", 100))
    if args.eval_episodes is None:
        args.eval_episodes = int(simulator_config.get("record_epo", 10)) if args.map_level is None else 0
    if args.map_level is not None and (
        min(args.train_episodes, args.eval_episodes) < 0 or not 1 <= args.train_episodes + args.eval_episodes <= 1000
    ):
        parser.error("Shared maps require 1–1000 total training and evaluation episodes")

    set_seed(args.seed)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_dir = f"logs/tensorboard_{timestamp}"
    writer = None if args.no_tensorboard else SummaryWriter(log_dir=tensorboard_dir)
    env = make_env(simulator_config, map_level=args.map_level)
    if args.map_level is not None:
        sys.path.append(str(Path(__file__).resolve().parents[2]))
        from baselines.manuscript_scene_adapter import reset_manuscript_scene

        env.shared_scene_reset = partial(reset_manuscript_scene, env, args.map_level)
    uav = env.Agents[0]
    if args.batch_size is not None:
        uav.Trainer.Batch_Size = args.batch_size
    rows = []
    start = time.time()
    for episode in tqdm(range(args.train_episodes), desc="DDPG-APF train"):
        map_episode = episode + 1 if args.map_level is not None else None
        row = run_episode(env, "ddpg", args.seed + episode, eps=args.eps, train=True, map_episode=map_episode)
        rows.append(row)
        if writer is not None:
            log_train_row(writer, row, episode + 1)

    for episode in tqdm(range(args.eval_episodes), desc="APF compare"):
        eval_seed = args.seed + 10000 + episode
        map_episode = args.train_episodes + episode + 1 if args.map_level is not None else None
        baseline_row = run_episode(env, "baseline", eval_seed, eps=0.0, train=False,
                                   render_html=not args.no_render_html, map_episode=map_episode)
        ddpg_row = run_episode(env, "ddpg", eval_seed, eps=0.0, train=False,
                               render_html=not args.no_render_html, map_episode=map_episode)
        rows.append(baseline_row)
        rows.append(ddpg_row)
    baseline = summarize(rows, "baseline")
    ddpg = summarize([row for row in rows if row["seed"] >= args.seed + 10000], "ddpg")
    reduction = 0.0
    if baseline["path_len"] > 0:
        reduction = (baseline["path_len"] - ddpg["path_len"]) / baseline["path_len"] * 100.0

    output_path = f"logs/ddpg_apf_experiment_{timestamp}.csv"
    with open(output_path, "w", newline="") as f:
        csv_writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "seed",
                "html_path",
                "map_len",
                "map_width",
                "map_height",
                "obstacle_count",
                "fixed_obstacle_height",
                "info",
                "success",
                "path_len",
                "steps",
                "score",
                "loss",
                "k_att",
                "k_rep",
            ] + (["map_level", "map_episode", "map_seed"] if args.map_level is not None else []),
        )
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    if writer is not None:
        writer.flush()
        writer.close()

    print("output", output_path)
    if not args.no_render_html:
        print("html_dir", "DataBase/experience")
    if not args.no_tensorboard:
        print("tensorboard_dir", tensorboard_dir)
    print("map", f"{env.len}x{env.width}x{env.h}", "path_len_units", "map_coordinate_units")
    fixed_height = getattr(env, "fixed_obstacle_height", "") if args.map_level is None else "per-obstacle heights"
    print("obstacle_count", len(env.buildings), "fixed_obstacle_height", fixed_height)
    print("train_episodes", args.train_episodes, "eval_episodes", args.eval_episodes)
    print("baseline_success_rate", round(baseline["success_rate"], 4), "baseline_path_len", round(baseline["path_len"], 4))
    print("ddpg_success_rate", round(ddpg["success_rate"], 4), "ddpg_path_len", round(ddpg["path_len"], 4))
    print("path_reduction_percent", round(reduction, 4))
    print("elapsed_sec", round(time.time() - start, 2))


if __name__ == "__main__":
    main()
