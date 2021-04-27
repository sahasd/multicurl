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
import os
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
from array2gif import write_gif
import torch
from collections import defaultdict
import numpy as np

from pettingzoo.butterfly import cooperative_pong_v2 as cooperative_pong
import supersuit as ss


def set_dqn_mode(dqns, mode="train"):
    for agent in dqns.keys():
        if mode == "train":
            dqns[agent].train()
        elif mode == "eval":
            dqns[agent].eval()
        else:
            raise Error("invalid mode specified")


def test_multi_agent_dqn(
    args, T, dqns, val_mems, metrics, results_dir, evaluate=False
):
    # init env for testing
    env = cooperative_pong.env(
        ball_speed=9, 
        left_paddle_speed=12,
        right_paddle_speed=12,
        cake_paddle=False,
        max_cycles=3000, 
        bounce_randomness=False
    )
    env = ss.color_reduction_v0(env, mode="full")
    env = ss.resize_v0(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.frame_skip_v0(env, 4)
    env = ss.dtype_v0(env,np.float32)
    env = ss.normalize_obs_v0(env)
    env = ss.clip_reward_v0(env, -1, 1)
    env.reset()

    metrics["steps"].append(T)
    T_rewards, T_Qs = {agent: [] for agent in env.agents}, {agent: [] for agent in env.agents}

    obs_list = {agent: [] for agent in env.agents}
    best_total_reward = -float("inf")

    for _ in range(args.evaluation_episodes):
        env.reset()
        reward_sum = defaultdict(lambda: 0)
        curr_obs_list = {agent: [] for agent in env.agents}
        for agent in env.agent_iter():
            observation, reward, done, _ = env.last()
            action = dqns[agent].act_e_greedy(torch.tensor(observation)) if not done else None       
            env.step(action)
            reward_sum[agent] += reward

            # add frames to list for gameplay gif generation    
            temp_obs = observation * 255
            for x in np.transpose(temp_obs,(2,0,1)):
                curr_obs_list[agent].append(np.stack((x,)*3,axis=0))

        for agent, agent_reward in reward_sum.items():
            T_rewards[agent].append(agent_reward)

        env.reset()

        if max(reward_sum.values()) > best_total_reward:
            best_total_reward =  max(reward_sum.values())
            for agent in env.agents:
                obs_list[agent] = curr_obs_list[agent]            
    env.reset()


    for agent in env.agents:
        for obs in val_mems[agent]:
            T_Qs[agent].append(dqns[agent].evaluate_q(torch.tensor(obs)))

    avg_reward = {}
    avg_Q = {}
    for agent in env.agents:
        avg_reward[agent] = sum(T_rewards[agent]) / len(T_rewards[agent])
        avg_Q[agent] = sum(T_Qs[agent]) / len(T_Qs[agent])

    if not evaluate:
        for agent in env.agents:
            # save model params if improved
            if avg_reward[agent] > metrics["best_avg_reward"][agent]:
                metrics["best_avg_reward"][agent] = avg_reward[agent]
                dqns[agent].save(results_dir)

            # Append to results and save metrics
            metrics["rewards"][agent].append(T_rewards[agent])
            metrics["Qs"][agent].append(T_Qs[agent])
            torch.save(metrics, os.path.join(results_dir, "metrics.pth"))

        # Plot
        for agent in env.agents:
            _plot_line(
                metrics["steps"],
                metrics["rewards"][agent],
                f"{agent} Reward",
                path=results_dir,
            )
            _plot_line(
                metrics["steps"], metrics["Qs"][agent], f"{agent} Q", path=results_dir
            )

            # Save best run as gif
            write_gif(obs_list[agent], f"{results_dir}/{agent}-{T}.gif", fps=15)

    return avg_reward, avg_Q


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=""):
    max_colour, mean_colour, std_colour, transparent = (
        "rgb(0, 132, 180)",
        "rgb(0, 172, 237)",
        "rgba(29, 202, 255, 0.2)",
        "rgba(0, 0, 0, 0)",
    )

    ys = torch.tensor(ys_population, dtype=torch.float32)
    ys_min, ys_max, ys_mean, ys_std = (
        ys.min(1)[0].squeeze(),
        ys.max(1)[0].squeeze(),
        ys.mean(1).squeeze(),
        ys.std(1).squeeze(),
    )
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(
        x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash="dash"), name="Max"
    )
    trace_upper = Scatter(
        x=xs,
        y=ys_upper.numpy(),
        line=Line(color=transparent),
        name="+1 Std. Dev.",
        showlegend=False,
    )
    trace_mean = Scatter(
        x=xs,
        y=ys_mean.numpy(),
        fill="tonexty",
        fillcolor=std_colour,
        line=Line(color=mean_colour),
        name="Mean",
    )
    trace_lower = Scatter(
        x=xs,
        y=ys_lower.numpy(),
        fill="tonexty",
        fillcolor=std_colour,
        line=Line(color=transparent),
        name="-1 Std. Dev.",
        showlegend=False,
    )
    trace_min = Scatter(
        x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash="dash"), name="Min"
    )

    plotly.offline.plot(
        {
            "data": [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
            "layout": dict(
                title=title, xaxis={"title": "Step"}, yaxis={"title": title}
            ),
        },
        filename=os.path.join(path, title + ".html"),
        auto_open=False,
    )
