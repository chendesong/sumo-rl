"""
Fair-TSC main training loop — paper Algorithm 1.

Stage 1: UE warm-up (T_warm steps under selfish actor π_UE) → updates π_UE, V^UE
Stage 2: Freeze π_UE, V^UE; MARL training under π_θ with fair advantage
         → updates π_θ, V^MARL, dual variables (λ^p_i, λ^s_i, µ)
"""

import csv
import os
import time
from typing import Dict, List

import numpy as np
import torch

from sumo_env import FairTSCEnv
from networks import SharedActor, SharedCritic
from rollout_buffer import RolloutBuffer
from ppo_core import ppo_update, bootstrap_last_values
from lagrangian import (
    compute_sacrifice_gaps, reshape_deltas_to_step_agent,
    theil_t_batch, LagrangianMultipliers,
)
import config as C


def collect_one_episode(env, actor, critic, buffer, device, seed=None):
    obs = env.reset(seed=seed)
    done = False
    ep_R = {a: 0.0 for a in env.agent_ids}
    ep_Cp = {a: 0.0 for a in env.agent_ids}
    ep_Cs = {a: 0.0 for a in env.agent_ids}
    n = 0
    while not done:
        g = env.get_global_obs(obs)
        local_b  = torch.from_numpy(np.stack([obs[a] for a in env.agent_ids])).to(device)
        global_b = torch.from_numpy(np.tile(g, (env.num_agents, 1))).to(device)
        idx_b    = torch.arange(env.num_agents, device=device)
        with torch.no_grad():
            action, logprob = actor.act(local_b, idx_b)
            value = critic(global_b, idx_b)
        action_dict = {a: int(action[i].item()) for i, a in enumerate(env.agent_ids)}
        next_obs, R, Cp, Cs, done, _ = env.step(action_dict)
        for i, a in enumerate(env.agent_ids):
            buffer.add(
                agent_id=a, local_obs=obs[a], global_obs=g,
                action=int(action[i].item()), logprob=float(logprob[i].item()),
                reward=R[a], value=float(value[i].item()), done=done,
                c_p=Cp[a], c_s=Cs[a],
            )
            ep_R[a]  += R[a]
            ep_Cp[a] += Cp[a]
            ep_Cs[a] += Cs[a]
        obs = next_obs
        n += 1
    last_v = bootstrap_last_values(critic, env.get_global_obs(obs),
                                   env.agent_ids, env.num_agents, device)
    buffer.compute_gae(last_v, gamma=C.GAMMA, gae_lambda=C.GAE_LAMBDA, device=device)
    return ep_R, ep_Cp, ep_Cs, n


def per_agent_batch_means(buffer, agent_ids):
    a_np = buffer.flat["agent_idx"].cpu().numpy()
    cp = buffer.flat["c_p"].cpu().numpy()
    cs = buffer.flat["c_s"].cpu().numpy()
    mp, ms = {}, {}
    for n, a in enumerate(agent_ids):
        mask = a_np == n
        mp[a] = float(cp[mask].mean()) if mask.any() else 0.0
        ms[a] = float(cs[mask].mean()) if mask.any() else 0.0
    return mp, ms


def main():
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    os.makedirs(C.CKPT_DIR, exist_ok=True)
    log_path = os.path.join(C.OUTPUT_DIR, "train_log.csv")
    print(f"output dir : {C.OUTPUT_DIR}")
    print(f"ckpt dir   : {C.CKPT_DIR}")
    print(f"log file   : {log_path}")

    env = FairTSCEnv(
        net_file=C.NET_FILE, route_file=C.ROUTE_FILE,
        out_csv_name=os.path.join(C.OUTPUT_DIR, "ep"),
        num_seconds=C.NUM_SECONDS, delta_time=C.DELTA_TIME, min_green=C.MIN_GREEN,
    )
    print(f"agents={env.agent_ids}  N={env.num_agents}  D_l={env.local_obs_dim}  D_g={env.global_obs_dim}  A={env.action_dim}")

    actor_marl  = SharedActor (env.local_obs_dim,  env.num_agents, env.action_dim, C.ACTOR_HIDDEN ).to(device)
    critic_marl = SharedCritic(env.global_obs_dim, env.num_agents,                  C.CRITIC_HIDDEN).to(device)
    actor_ue    = SharedActor (env.local_obs_dim,  env.num_agents, env.action_dim, C.ACTOR_HIDDEN ).to(device)
    critic_ue   = SharedCritic(env.global_obs_dim, env.num_agents,                  C.CRITIC_HIDDEN).to(device)

    om_a  = torch.optim.Adam(actor_marl.parameters(),  lr=C.ACTOR_LR)
    om_c  = torch.optim.Adam(critic_marl.parameters(), lr=C.CRITIC_LR)
    ou_a  = torch.optim.Adam(actor_ue.parameters(),    lr=C.ACTOR_LR)
    ou_c  = torch.optim.Adam(critic_ue.parameters(),   lr=C.CRITIC_LR)

    lagr = LagrangianMultipliers(
        agent_ids=env.agent_ids, d_p=C.D_P, d_s=C.D_S, t_max=C.T_MAX,
        eta_lambda=C.ETA_LAMBDA, eta_mu=C.ETA_MU,
    )

    log_fields = [
        "stage","episode","global_step","wall_time_s",
        "reward_mean","reward_min","reward_max",
    ] + [f"reward_{a}" for a in env.agent_ids] + [
        "C_p_mean","C_s_mean",
        "delta_mean","delta_max","theil",
        "lambda_p_mean","lambda_p_max","lambda_s_mean","lambda_s_max","mu",
        "policy_loss","value_loss","entropy","approx_kl","explained_var",
    ]
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()

    t0 = time.time()
    gstep = 0
    ep = 0
    aid_map = {i: a for i, a in enumerate(env.agent_ids)}

    # ── STAGE 1 ─────────────────────────────────────────────────
    print(f"\n{'='*70}\nSTAGE 1: UE warm-up   (target = {C.T_WARM} steps)\n{'='*70}")
    while gstep < C.T_WARM:
        buf = RolloutBuffer(env.agent_ids, env.num_agents)
        ep_R, ep_Cp, ep_Cs, n = collect_one_episode(
            env, actor_ue, critic_ue, buf, device, seed=C.SEED + ep,
        )
        gstep += n
        ep += 1
        st = ppo_update(
            actor=actor_ue, critic=critic_ue,
            actor_optim=ou_a, critic_optim=ou_c, buffer=buf,
            ppo_epochs=C.PPO_EPOCHS, minibatch_size=C.MINIBATCH_SIZE,
            clip_eps=C.CLIP_EPS, entropy_coeff=C.ENTROPY_COEFF,
            vf_coeff=C.VF_COEFF, grad_clip=C.GRAD_CLIP,
        )
        rR = np.array(list(ep_R.values()))
        et = time.time() - t0
        print(f"[STAGE1] ep={ep:3d} step={gstep:6d}/{C.T_WARM} "
              f"R̄={rR.mean():+.1f} ploss={st['policy_loss']:+.4f} "
              f"vloss={st['value_loss']:.0f} H={st['entropy']:.3f} t={et:.0f}s")
        log_writer.writerow({
            "stage":1,"episode":ep,"global_step":gstep,"wall_time_s":et,
            "reward_mean":float(rR.mean()),"reward_min":float(rR.min()),"reward_max":float(rR.max()),
            **{f"reward_{a}":float(ep_R[a]) for a in env.agent_ids},
            "C_p_mean":float(np.mean(list(ep_Cp.values()))),
            "C_s_mean":float(np.mean(list(ep_Cs.values()))),
            "delta_mean":0.0,"delta_max":0.0,"theil":0.0,
            "lambda_p_mean":0.0,"lambda_p_max":0.0,"lambda_s_mean":0.0,"lambda_s_max":0.0,"mu":0.0,
            "policy_loss":st["policy_loss"],"value_loss":st["value_loss"],
            "entropy":st["entropy"],"approx_kl":st["approx_kl"],"explained_var":st["explained_var"],
        })
        log_file.flush()

    # Freeze UE
    for p in actor_ue.parameters():  p.requires_grad = False
    for p in critic_ue.parameters(): p.requires_grad = False
    actor_ue.eval(); critic_ue.eval()
    print(f"\n[STAGE1 done] gstep={gstep}, π_UE and V^UE frozen.\n")

    # ── STAGE 2 ─────────────────────────────────────────────────
    print(f"{'='*70}\nSTAGE 2: MARL training   (target = {C.TOTAL_STEPS} total steps)\n{'='*70}")
    while gstep < C.TOTAL_STEPS:
        buf = RolloutBuffer(env.agent_ids, env.num_agents)
        ep_R, ep_Cp, ep_Cs, n = collect_one_episode(
            env, actor_marl, critic_marl, buf, device, seed=C.SEED + ep,
        )
        gstep += n
        ep += 1

        deltas = compute_sacrifice_gaps(buf, critic_ue, critic_marl, device)
        deltas_TN = reshape_deltas_to_step_agent(deltas, buf.flat["agent_idx"], env.num_agents)
        theil = theil_t_batch(deltas_TN, eps=C.THEIL_EPS)

        mp, ms = per_agent_batch_means(buf, env.agent_ids)
        lagr.update_intra(mp, ms)
        lagr.update_inter(theil)

        lagr.apply_fair_advantage(buf, deltas, aid_map)

        st = ppo_update(
            actor=actor_marl, critic=critic_marl,
            actor_optim=om_a, critic_optim=om_c, buffer=buf,
            ppo_epochs=C.PPO_EPOCHS, minibatch_size=C.MINIBATCH_SIZE,
            clip_eps=C.CLIP_EPS, entropy_coeff=C.ENTROPY_COEFF,
            vf_coeff=C.VF_COEFF, grad_clip=C.GRAD_CLIP,
        )

        rR = np.array(list(ep_R.values()))
        ls = lagr.stats()
        et = time.time() - t0
        print(f"[STAGE2] ep={ep:3d} step={gstep:6d}/{C.TOTAL_STEPS} "
              f"R̄={rR.mean():+.1f} "
              f"theil={theil:.4f} λp̄={ls['lambda_p_mean']:.3f} "
              f"λs̄={ls['lambda_s_mean']:.3f} µ={ls['mu']:.4f} "
              f"H={st['entropy']:.3f} t={et:.0f}s")

        log_writer.writerow({
            "stage":2,"episode":ep,"global_step":gstep,"wall_time_s":et,
            "reward_mean":float(rR.mean()),"reward_min":float(rR.min()),"reward_max":float(rR.max()),
            **{f"reward_{a}":float(ep_R[a]) for a in env.agent_ids},
            "C_p_mean":float(np.mean(list(ep_Cp.values()))),
            "C_s_mean":float(np.mean(list(ep_Cs.values()))),
            "delta_mean":float(deltas.mean().item()),"delta_max":float(deltas.max().item()),"theil":float(theil),
            "lambda_p_mean":ls["lambda_p_mean"],"lambda_p_max":ls["lambda_p_max"],
            "lambda_s_mean":ls["lambda_s_mean"],"lambda_s_max":ls["lambda_s_max"],"mu":ls["mu"],
            "policy_loss":st["policy_loss"],"value_loss":st["value_loss"],
            "entropy":st["entropy"],"approx_kl":st["approx_kl"],"explained_var":st["explained_var"],
        })
        log_file.flush()

        if ep % C.SAVE_CKPT_EVERY_N == 0:
            cp = os.path.join(C.CKPT_DIR, f"ep_{ep:04d}.pt")
            torch.save({
                "episode":ep,"global_step":gstep,
                "actor_marl":actor_marl.state_dict(),"critic_marl":critic_marl.state_dict(),
                "actor_ue":actor_ue.state_dict(),"critic_ue":critic_ue.state_dict(),
                "lambda_p":lagr.lambda_p,"lambda_s":lagr.lambda_s,"mu":lagr.mu,
            }, cp)
            print(f"   [ckpt] saved {cp}")

    cp = os.path.join(C.CKPT_DIR, "final.pt")
    torch.save({
        "episode":ep,"global_step":gstep,
        "actor_marl":actor_marl.state_dict(),"critic_marl":critic_marl.state_dict(),
        "actor_ue":actor_ue.state_dict(),"critic_ue":critic_ue.state_dict(),
        "lambda_p":lagr.lambda_p,"lambda_s":lagr.lambda_s,"mu":lagr.mu,
    }, cp)
    print(f"\n[done] {ep} episodes, {gstep} steps, {(time.time()-t0)/60:.1f} min. Final ckpt: {cp}")

    log_file.close()
    env.close()


if __name__ == "__main__":
    main()