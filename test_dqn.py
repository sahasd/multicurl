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


from env import Env


def set_dqn_mode(dqns, mode="train"):
    for agent in dqns.keys():
        if mode == "train":
            dqns[agent].train()
        elif mode == "eval":
            dqns[agent].eval()
        else:
            raise Error("invalid mode specified")


def test_multi_agent_dqn(
    args, env, T, dqns, val_mems, metrics, results_dir, evaluate=False
):
    metrics["steps"].append(T)
    T_rewards, T_Qs = defaultdict(lambda: []), defaultdict(lambda: [])

    obs_list = []
    best_total_reward = -9999

    for _ in range(args.evaluation_episodes):
        env.reset()
        reward_sum = defaultdict(lambda: 0)
        curr_obs_list = []
        for agent in env.agent_iter():
            observation, reward, done, _ = env.last()
            if done == True:
                continue
            action = dqns[agent].act_e_greedy(observation)

            env.step(action)
            reward_sum[agent] += reward
        total_reward = 0
        for agent, agent_reward in reward_sum.items():
            T_rewards[agent].append(agent_reward)
            total_reward += agent_reward

        if total_reward > best_total_reward:
            best_total_reward = total_reward
            obs_list = curr_obs_list

    for agent in env.agents():
        for obs in val_mems[agent]:
            T_Qs[agent].append(dqns[agent].evaluate_q(obs))

    avg_reward = {}
    avg_Q = {}
    for agent in env.agents():
        avg_reward[agent] = sum(T_rewards[agent]) / len(T_rewards[agent])
        avg_Q[agent] = sum(T_Qs[agent]) / len(T_Qs[agent])

    if not evaluate:
        for agent in env.agents():
            # save model params if improved
            if avg_reward[agent] > metrics["best_avg_reward"][agent]:
                metrics["best_avg_reward"][agent] = avg_reward[agent]
                dqns[agent].save(results_dir)

            # Append to results and save metrics
            metrics["rewards"].append(T_rewards)
            metrics["Qs"].append(T_Qs)
            torch.save(metrics, os.path.join(results_dir, "metrics.pth"))

        # Plot
        for agent in env.agents():
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
        write_gif(obs_list, f"{T}.gif", fps=15)
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
