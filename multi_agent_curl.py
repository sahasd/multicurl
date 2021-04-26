# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2017 Kai Arulkumaran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# ==============================================================================
from __future__ import division
import argparse
import bz2
from datetime import datetime
import os
import atari_py
import numpy as np
import torch
import tqdm

from agent import Agent
from env import Env
from memory import ReplayMemory
from test_dqn import test_multi_agent_dqn, set_dqn_mode

import supersuit as ss
from pettingzoo.butterfly import cooperative_pong_v2 as cooperative_pong

def run(worskpace_dir):
    seed = np.random.randint(12345)
    # Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
    parser = argparse.ArgumentParser(description="Rainbow")
    parser.add_argument("--id", type=str, default="default", help="Experiment ID")
    parser.add_argument("--seed", type=int, default=seed, help="Random seed")
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--game",
        type=str,
        default="ms_pacman",
        choices=atari_py.list_games(),
        help="ATARI game",
    )
    parser.add_argument(
        "--T-max",
        type=int,
        default=int(1e5),
        metavar="STEPS",
        help="Number of training steps (4x number of frames)",
    )
    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=int(108e3),
        metavar="LENGTH",
        help="Max episode length in game frames (0 to disable)",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=4,
        metavar="T",
        help="Number of consecutive states processed",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="data-efficient",
        choices=["canonical", "data-efficient"],
        metavar="ARCH",
        help="Network architecture",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        metavar="SIZE",
        help="Network hidden size",
    )
    parser.add_argument(
        "--noisy-std",
        type=float,
        default=0.1,
        metavar="σ",
        help="Initial standard deviation of noisy linear layers",
    )
    parser.add_argument(
        "--atoms",
        type=int,
        default=51,
        metavar="C",
        help="Discretised size of value distribution",
    )
    parser.add_argument(
        "--V-min",
        type=float,
        default=-10,
        metavar="V",
        help="Minimum of value distribution support",
    )
    parser.add_argument(
        "--V-max",
        type=float,
        default=10,
        metavar="V",
        help="Maximum of value distribution support",
    )
    parser.add_argument(
        "--model", type=str, metavar="PARAMS", help="Pretrained model (state dict)"
    )
    parser.add_argument(
        "--memory-capacity",
        type=int,
        default=int(1e5),
        metavar="CAPACITY",
        help="Experience replay memory capacity",
    )
    parser.add_argument(
        "--replay-frequency",
        type=int,
        default=1,
        metavar="k",
        help="Frequency of sampling from memory",
    )
    parser.add_argument(
        "--priority-exponent",
        type=float,
        default=0.5,
        metavar="ω",
        help="Prioritised experience replay exponent (originally denoted α)",
    )
    parser.add_argument(
        "--priority-weight",
        type=float,
        default=0.4,
        metavar="β",
        help="Initial prioritised experience replay importance sampling weight",
    )
    parser.add_argument(
        "--multi-step",
        type=int,
        default=20,
        metavar="n",
        help="Number of steps for multi-step return",
    )
    parser.add_argument(
        "--discount", type=float, default=0.99, metavar="γ", help="Discount factor"
    )
    parser.add_argument(
        "--target-update",
        type=int,
        default=int(2e3),
        metavar="τ",
        help="Number of steps after which to update target network",
    )
    parser.add_argument(
        "--reward-clip",
        type=int,
        default=1,
        metavar="VALUE",
        help="Reward clipping (0 to disable)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.0001, metavar="η", help="Learning rate"
    )
    parser.add_argument(
        "--adam-eps", type=float, default=1.5e-4, metavar="ε", help="Adam epsilon"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, metavar="SIZE", help="Batch size"
    )
    parser.add_argument(
        "--norm-clip",
        type=float,
        default=10,
        metavar="NORM",
        help="Max L2 norm for gradient clipping",
    )
    parser.add_argument(
        "--learn-start",
        type=int,
        default=int(1600),
        metavar="STEPS",
        help="Number of steps before starting training",
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate only")
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=10000,
        metavar="STEPS",
        help="Number of training steps between evaluations",
    )
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=10,
        metavar="N",
        help="Number of evaluation episodes to average over",
    )
    # TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
    parser.add_argument(
        "--evaluation-size",
        type=int,
        default=500,
        metavar="N",
        help="Number of transitions to use for validating Q",
    )
    parser.add_argument(
        "--render", action="store_true", help="Display screen (testing only)"
    )
    parser.add_argument(
        "--enable-cudnn",
        action="store_true",
        help="Enable cuDNN (faster but nondeterministic)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        default=10000,
        help="How often to checkpoint the model, defaults to 0 (never checkpoint)",
    )
    parser.add_argument("--memory", help="Path to save/load the memory from")
    parser.add_argument(
        "--disable-bzip-memory",
        action="store_true",
        help="Don't zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)",
    )
    # Setup
    args = parser.parse_args(["--game", "ms_pacman", "--enable-cudnn"])
    print(f"ARGS: {args}")
    xid = "curl-" + args.game + "-" + str(seed)
    args.id = xid

    print(" " * 26 + "Options")
    for k, v in vars(args).items():
        print(" " * 26 + k + ": " + str(v))
    results_dir = os.path.join(worskpace_dir, "results", args.id)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    metrics = {"steps": [], "rewards": [], "Qs": [], "best_avg_reward": -float("inf")}
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device("cuda")
        print(args.device)
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        torch.backends.cudnn.enabled = args.enable_cudnn
    else:
        args.device = torch.device("cpu")

    # Simple ISO 8601 timestamped logger
    def log(s):
        print("[" + str(datetime.now().strftime("%Y-%m-%dT%H:%M:%S")) + "] " + s)

    def load_memory(memory_path, disable_bzip):
        if disable_bzip:
            with open(memory_path, "rb") as pickle_file:
                return pickle.load(pickle_file)
        else:
            with bz2.open(memory_path, "rb") as zipped_pickle_file:
                return pickle.load(zipped_pickle_file)

    def save_memory(memory, memory_path, disable_bzip):
        if disable_bzip:
            with open(memory_path, "wb") as pickle_file:
                pickle.dump(memory, pickle_file)
        else:
            with bz2.open(memory_path, "wb") as zipped_pickle_file:
                pickle.dump(memory, zipped_pickle_file)

    env = cooperative_pong.env(
        ball_speed=9, 
        left_paddle_speed=12,
        right_paddle_speed=12,
        cake_paddle=False,
        max_cycles=900, 
        bounce_randomness=False
    )
    env = ss.color_reduction_v0(env, mode="full")
    env = ss.resize_v0(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.frame_skip_v0(env, 4)
    env = ss.dtype_v0(env,np.float32)
    env = ss.normalize_obs_v0(env)

    dqns = {}
    mems = {}
    val_mems = {}

    # Agents
    env.reset()
    for agent in env.agents:
        dqns[agent] = Agent(args, env.action_spaces[agent].n)

        # If a model is provided, and evaluate is fale, presumably we want to resume, so try to load memory
        if args.model is not None and not args.evaluate:
            if not args.memory:
                raise ValueError(
                    "Cannot resume training without memory save path. Aborting..."
                )
            elif not os.path.exists(args.memory):
                raise ValueError(
                    "Could not find memory file at {path}. Aborting...".format(
                        path=args.memory
                    )
                )
            mems = load_memory(args.memory, args.disable_bzip_memory)
        else:
            mems[agent] = ReplayMemory(args, args.memory_capacity)

        priority_weight_increase = (1 - args.priority_weight) / (
            args.T_max - args.learn_start
        )

        # Construct validation memory
        val_mems[agent] = ReplayMemory(args, args.evaluation_size)

    T, converged = 0, False
    while not converged:
        i = 0
        env.reset()
        for agent in env.agent_iter(args.evaluation_size):
            if i % len(env.agents):
                T += 1
                if T > args.evaluation_size:
                    converged=True
                    break
            observation, reward, done, info = env.last()
            action = np.random.randint(0,env.action_spaces[agent].n) if not done else None
            env.step(action)
            if not done:
                val_mems[agent].append(torch.tensor(observation), None, None, done)
            i += 1

    if args.evaluate:
        set_dqn_mode(dqns, mode="eval")  # Set DQN (online network) to evaluation mode
        avg_reward, avg_Q = test_multi_agent_dqn(
            args, env, 0, dqns, val_mems, metrics, results_dir, evaluate=True
        )  # Test
        print("Avg. reward: " + str(avg_reward) + " | Avg. Q: " + str(avg_Q))
    else:
        set_dqn_mode(dqns, mode="train")
        T, converged = 0, False
        pbar = tqdm(total=args.T_max)
        while not converged:
            env.reset()
            i = 0
            for agent in env.agent_iter(args.T_max): 
                i += 1
                if i % len(env.agents):
                    pbar.update(1)
                    T += 1
                    if T > args.T_max:
                        converged = True
                        break
                if T % args.replay_frequency == 0:
                    dqns[agent].reset_noise()  # Draw a new set of noisy weights

                observation, reward, done, info = env.last()
                action = dqns[agent].act(torch.tensor(observation)) if not done else None # Choose an action greedily (with noisy weights)
                env.step(action)  # Step

                if args.reward_clip > 0:
                    reward = max(
                        min(reward, args.reward_clip), -args.reward_clip
                    )  # Clip rewards
                if not done:
                    mems[agent].append(
                    torch.tensor(observation), action, reward, done
                )  # Append transition to memory

                # Train and test
                if T >= args.learn_start:
                    # Anneal importance sampling weight β to 1
                    mems[agent].priority_weight = min(
                        mems[agent].priority_weight + priority_weight_increase, 1
                    )

                    if T % args.replay_frequency == 0:
                        # for _ in range(4):
                        dqns[agent].learn(
                            mems[agent]
                        )  # Train with n-step distributional double-Q learning
                        dqns[agent].update_momentum_net()  # MoCo momentum upate

                    if T % args.evaluation_interval == 0:
                        set_dqn_mode(dqns,mode="eval")
                        avg_reward, avg_Q = test_multi_agent_dqn(
                            args, env, T, dqns, val_mems, metrics, results_dir
                        )  # Test
                        log(
                            "T = "
                            + str(T)
                            + " / "
                            + str(args.T_max)
                            + " | Avg. reward: "
                            + str(avg_reward)
                            + " | Avg. Q: "
                            + str(avg_Q)
                        )
                        # Set DQNs (online network) back to training mode
                        set_dqn_mode(dqns, mode="train")

                    # If memory path provided, save it
                    if args.memory is not None:
                        save_memory(mems, args.memory, args.disable_bzip_memory)

                    # Update target network
                    if T % args.target_update == 0:
                        dqns[agent].update_target_net()

                    # Checkpoint the network
                    if (args.checkpoint_interval != 0) and (
                        T % args.checkpoint_interval == 0
                    ):
                        dqns[agent].save(results_dir, f"checkpoint-{agent}-{T}.pth")
        pbar.close()
run("./")
