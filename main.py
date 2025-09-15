#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CCOD + ns3-ai main (wersja pod równoległość, baseline i duże historie)
"""

from argparse import ArgumentParser
from ctypes import *
import os
import hashlib

import jax.numpy as jnp
import optax
from chex import Array
from flax import linen as nn

from ext import IEEE_802_11_CCOD
from py_interface import *

from reinforced_lib import RLib
from reinforced_lib.agents.deep import DDQN, DDPG
from reinforced_lib.agents.mab import SARSA

# ================== USTAWIENIA DRL ==================

INTERACTION_PERIOD = 1e-2
SIMULATION_TIME = 60
MAX_HISTORY_LENGTH = IEEE_802_11_CCOD.max_history_length
HISTORY_LENGTH = 300
THR_SCALE = 5 * 150 * INTERACTION_PERIOD * 10

DQN_LEARNING_RATE = 4e-4
DQN_EPSILON = 0.9
DQN_EPSILON_DECAY = 0.99991
DQN_EPSILON_MIN = 0.001

DDPG_Q_LEARNING_RATE = 4e-3
DDPG_A_LEARNING_RATE = 4e-4
DDPG_NOISE = 4.0
DDPG_NOISE_DECAY = 0.99994
DDPG_NOISE_MIN = 0.0271

REWARD_DISCOUNT = 0.7
LSTM_HIDDEN_SIZE = 8
SOFT_UPDATE = 4e-3

REPLAY_BUFFER_SIZE = 18000
REPLAY_BUFFER_BATCH_SIZE = 32
REPLAY_BUFFER_STEPS = 1

# ================== STRUKTURY ns3-ai ==================

class Env(Structure):
    _pack_ = 1
    _fields_ = [
        ('history', c_float * MAX_HISTORY_LENGTH),
        ('reward', c_float)
    ]

class Act(Structure):
    _pack_ = 1
    _fields_ = [
        ('action', c_float)
    ]

# Nazwa symulacji (możesz nadpisać ENV-em)
SIMULATION_NAME = os.getenv("NS3AI_SIM_NAME", "ccod-sim")

# ========= POMOCNICZE: rozmiar puli & mempoolKey =========

def _default_memory_size_bytes() -> int:
    """Domyślny rozmiar puli: max(8 MB, 8×(Env+Act))."""
    from ctypes import sizeof
    env_act = 8 * (sizeof(Env) + sizeof(Act))
    return max(8 * 1024 * 1024, env_act)

def resolve_memory_size() -> int:
    """Z ENV `NS3AI_MEM_MB` (MB) lub domyślny."""
    mb = os.getenv("NS3AI_MEM_MB", "").strip()
    if mb.isdigit() and int(mb) > 0:
        return int(mb) * 1024 * 1024
    return _default_memory_size_bytes()

def derive_mempool_key(cli_key: int | None, ns3_args: dict) -> int:
    """
    Zwróć mempoolKey:
      - jeśli podany w CLI i >0 → ten,
      - w przeciwnym razie deterministycznie z (RngRun, nWifi, scenario)
        oraz ENV: NS3AI_MEMKEY_BASE (domyślnie 1000), NS3AI_MEMKEY_RANGE (domyślnie 4000).
    """
    try:
        if cli_key is not None and int(cli_key) > 0:
            return int(cli_key)
    except Exception:
        pass

    base = int(os.getenv("NS3AI_MEMKEY_BASE", "1000"))
    rng  = int(os.getenv("NS3AI_MEMKEY_RANGE", "4000"))
    if rng <= 0:
        rng = 40000

    txt = f"{ns3_args.get('RngRun')}_{ns3_args.get('nWifi')}_{ns3_args.get('scenario','')}"
    h = int(hashlib.md5(txt.encode("utf-8")).hexdigest()[:6], 16) % rng
    return base + h

# ================== SIECI NN ==================

def add_batch_dim(x: Array, base_ndims: int) -> Array:
    if x.ndim == base_ndims and base_ndims > 1:
        return x[None, ...]
    elif x.ndim == base_ndims and base_ndims == 1:
        return x[..., None]
    else:
        return x

class DQNNetwork(nn.Module):
    @nn.compact
    def __call__(self, s: Array) -> Array:
        s = add_batch_dim(s, base_ndims=2)
        s = nn.RNN(nn.OptimizedLSTMCell(
            LSTM_HIDDEN_SIZE,
            activation_fn=nn.relu,
            kernel_init=nn.initializers.glorot_uniform()
        ))(s)[:, -1]
        s = nn.Dense(128, kernel_init=nn.initializers.glorot_uniform())(s)
        s = nn.relu(s)
        s = nn.Dense(64, kernel_init=nn.initializers.glorot_uniform())(s)
        s = nn.relu(s)
        return nn.Dense(7, kernel_init=nn.initializers.glorot_uniform())(s)

class DDPGQNetwork(nn.Module):
    @nn.compact
    def __call__(self, s: Array, a: Array) -> Array:
        s = add_batch_dim(s, base_ndims=2)
        s = nn.RNN(nn.OptimizedLSTMCell(
            LSTM_HIDDEN_SIZE,
            kernel_init=nn.initializers.uniform(1 / jnp.sqrt(LSTM_HIDDEN_SIZE)),
            carry_init=nn.initializers.normal(1.0)
        ))(s, init_key=self.make_rng('rlib'))[:, -1]
        s = nn.relu(s)
        a = add_batch_dim(a, base_ndims=1)
        x = jnp.concatenate([s, a], axis=1)
        x = nn.Dense(128, kernel_init=nn.initializers.variance_scaling(1 / 3, 'fan_in', 'uniform'))(x)
        x = nn.relu(x)
        x = nn.Dense(64, kernel_init=nn.initializers.variance_scaling(1 / 3, 'fan_in', 'uniform'))(x)
        x = nn.relu(x)
        return nn.Dense(1, kernel_init=nn.initializers.uniform(3e-3))(x)

class DDPGANetwork(nn.Module):
    @nn.compact
    def __call__(self, s: Array) -> Array:
        s = add_batch_dim(s, base_ndims=2)
        s = nn.RNN(nn.OptimizedLSTMCell(
            LSTM_HIDDEN_SIZE,
            kernel_init=nn.initializers.uniform(1 / jnp.sqrt(LSTM_HIDDEN_SIZE)),
            carry_init=nn.initializers.normal(1.0)
        ))(s, init_key=self.make_rng('rlib'))[:, -1]
        s = nn.relu(s)
        s = nn.Dense(128, kernel_init=nn.initializers.variance_scaling(1 / 3, 'fan_in', 'uniform'))(s)
        s = nn.relu(s)
        s = nn.Dense(64, kernel_init=nn.initializers.variance_scaling(1 / 3, 'fan_in', 'uniform'))(s)
        s = nn.relu(s)
        s = nn.Dense(1, kernel_init=nn.initializers.uniform(3e-3))(s).squeeze()
        return 3 * nn.tanh(s) + 3  # → [0, 6]

# ================== RUN ==================

def run(ns3_args, ns3_path, cli_mempool_key, agent_type, agent_params, rlib_args):
    csv_path = rlib_args['csv_path']
    csv_file = open(csv_path, 'w') if csv_path else None

    if not rlib_args['load_path']:
        rl = RLib(
            agent_type=agent_type,
            agent_params=agent_params,
            ext_type=IEEE_802_11_CCOD,
            ext_params={'history_length': ns3_args['historyLength']}
        )
        rl.init(rlib_args['seed'])
    else:
        rl = RLib.load(rlib_args['load_path'])

    mem_bytes = resolve_memory_size()
    mem_key   = derive_mempool_key(cli_mempool_key, ns3_args)
    show_out  = os.getenv("NS3AI_SHOW_OUTPUT", "1") not in ("0", "false", "False")

    ns3_args = dict(ns3_args)
    ns3_args["mempoolKey"] = mem_key

    exp = Experiment(mem_key, mem_bytes, SIMULATION_NAME, ns3_path, using_waf=False)
    var = Ns3AIRL(mem_key, Env, Act)

    baseline_mode = bool(ns3_args.get("dryRun", False))

    try:
        ns3_process = exp.run(ns3_args, show_output=show_out)
        step = 0

        while not var.isFinish():
            with var as data:
                if data is None:
                    break

                reward = float(data.env.reward)

                if baseline_mode:
                    if csv_file:
                        csv_file.write(
                            f"{agent_type.__name__},{ns3_args['scenario']},{ns3_args['nWifi']},{ns3_args['RngRun']},"
                            f"{step * INTERACTION_PERIOD},{reward * THR_SCALE}\n"
                        )
                    data.act.action = 0.0
                else:
                    observation = {'history': data.env.history, 'reward': reward}
                    action = rl.sample(**observation, is_training=rlib_args['is_training'])
                    data.act.action = float(action)

                    if csv_file:
                        csv_file.write(
                            f"{agent_type.__name__},{ns3_args['scenario']},{ns3_args['nWifi']},{ns3_args['RngRun']},"
                            f"{step * INTERACTION_PERIOD},{reward * THR_SCALE}\n"
                        )

                step += 1

        if rlib_args['is_training'] and rlib_args['save_path'] and not baseline_mode:
            rl.save(agent_ids=0, path=rlib_args['save_path'])

        ns3_process.wait()

    finally:
        del exp
        if csv_file:
            csv_file.close()
        try:
            if ns3_process.poll() is None:
                ns3_process.terminate()
        except Exception:
            pass

# ================== MAIN ==================

if __name__ == '__main__':
    ap = ArgumentParser()

    # Python args
    ap.add_argument('--agent', default='DDQN', type=str)
    ap.add_argument('--loadPath', default='', type=str)
    ap.add_argument('--mempoolKey', default=0, type=int, help=">0 aby wymusić konkretny klucz puli")
    ap.add_argument('--ns3Path', required=True, type=str)
    ap.add_argument('--pythonSeed', default=42, type=int)
    ap.add_argument('--sampleOnly', default=False, action='store_true')
    ap.add_argument('--savePath', default='', type=str)
    ap.add_argument('--csvPath', default='', type=str)

    # ns-3 args
    ap.add_argument('--agentType', default='discrete', type=str)
    ap.add_argument('--CW', default=0, type=int)
    ap.add_argument('--dryRun', default=False, action='store_true')           # baseline domyślnie ON
    ap.add_argument('--envStepTime', default=INTERACTION_PERIOD, type=float)
    ap.add_argument('--historyLength', default=HISTORY_LENGTH, type=int)
    ap.add_argument('--nonZeroStart', default=True, action='store_true')
    ap.add_argument('--nWifi', default=55, type=int)
    ap.add_argument('--scenario', default='convergence', type=str)
    ap.add_argument('--seed', default=42, type=int)
    ap.add_argument('--simTime', default=SIMULATION_TIME, type=float)
    ap.add_argument('--tracing', default=False, action='store_true')
    ap.add_argument('--verbose', default=False, action='store_true')

    args = vars(ap.parse_args())

    assert args['historyLength'] <= MAX_HISTORY_LENGTH, \
        f"HISTORY_LENGTH={args['historyLength']} exceeded MAX_HISTORY_LENGTH={MAX_HISTORY_LENGTH}!"

    args['RngRun'] = args.pop('seed')
    agent = args.pop('agent')

    agent_type = {
        'DDQN': DDQN,
        'DDPG': DDPG,
        'SARSA': SARSA
    }

    default_params = {
        'DDQN': {
            'q_network': DQNNetwork(),
            'optimizer': optax.adam(DQN_LEARNING_RATE),
            'experience_replay_buffer_size': REPLAY_BUFFER_SIZE,
            'experience_replay_batch_size': REPLAY_BUFFER_BATCH_SIZE,
            'experience_replay_steps': REPLAY_BUFFER_STEPS,
            'discount': REWARD_DISCOUNT,
            'epsilon': DQN_EPSILON,
            'epsilon_decay': DQN_EPSILON_DECAY,
            'epsilon_min': DQN_EPSILON_MIN,
            'tau': SOFT_UPDATE
        },
        'DDPG': {
            'a_network': DDPGANetwork(),
            'a_optimizer': optax.adam(DDPG_A_LEARNING_RATE),
            'q_network': DDPGQNetwork(),
            'q_optimizer': optax.adam(DDPG_Q_LEARNING_RATE),
            'experience_replay_buffer_size': REPLAY_BUFFER_SIZE,
            'experience_replay_batch_size': REPLAY_BUFFER_BATCH_SIZE,
            'experience_replay_steps': REPLAY_BUFFER_STEPS,
            'discount': REWARD_DISCOUNT,
            'noise': DDPG_NOISE,
            'noise_decay': DDPG_NOISE_DECAY,
            'noise_min': DDPG_NOISE_MIN,
            'tau': SOFT_UPDATE
        },
        'SARSA': {
            'alpha': float(os.getenv("SARSA_ALPHA", 0.089)),
            'gamma': float(os.getenv("SARSA_GAMMA", 0.239)),
            'epsilon': float(os.getenv("SARSA_EPSILON", 0.1)),
            'state_space_size': 64,
            'action_space_size': 7,
            'use_fixed_epsilon': bool(int(os.getenv("SARSA_FIXED_EPS", 1)))
        }
    }

    if agent == "SARSA":
        print(f"use_fixed_epsilon = {default_params['SARSA']['use_fixed_epsilon']}")

    rlib_args = {
        'seed': args.pop('pythonSeed'),
        'is_training': not args.pop('sampleOnly'),
        'load_path': args.pop('loadPath'),
        'save_path': args.pop('savePath'),
        'csv_path': args.pop('csvPath')
    }

    run(
        args,
        args.pop('ns3Path'),
        args.pop('mempoolKey'),
        agent_type[agent],
        default_params[agent],
        rlib_args,
    )
