from functools import partial
import gymnasium as gym
import jax
import jax.numpy as jnp
from chex import dataclass, Array, Scalar, PRNGKey
from reinforced_lib.agents import BaseAgent, AgentState


@dataclass
class SarsaState(AgentState):
    Q: Array
    epsilon: Scalar
    prev_state: int
    prev_action: int
    prev_reward: float
    timestep: int

class SARSA(BaseAgent):
    def __init__(
        self,
        alpha: float,
        gamma: float,
        epsilon: float,
        state_space_size: int,
        action_space_size: int = 7,
        use_fixed_epsilon: bool = True
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.initial_epsilon = epsilon
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.use_fixed_epsilon = use_fixed_epsilon

        self.update = jax.jit(partial(self.update, alpha=alpha, gamma=gamma, use_fixed_epsilon=use_fixed_epsilon))
        self.sample = jax.jit(partial(self.sample, use_fixed_epsilon=use_fixed_epsilon))

    def init(self, key: PRNGKey) -> SarsaState:
        return SarsaState(
            Q=jnp.zeros((self.state_space_size, self.action_space_size)),
            epsilon=self.initial_epsilon,
            prev_state=0,
            prev_action=0,
            prev_reward=0.0,
            timestep=0
        )

    @property
    def update_observation_space(self):
        return gym.spaces.Dict({
            'current_state': gym.spaces.Discrete(self.state_space_size),
            'action': gym.spaces.Discrete(self.action_space_size),
            'reward': gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float)
        })

    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'current_state': gym.spaces.Discrete(self.state_space_size)
        })

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.action_space_size)

    @staticmethod
    def update(
        state: SarsaState,
        key: PRNGKey,
        current_state: int,
        action: int,
        reward: float,
        alpha,
        gamma,
        use_fixed_epsilon: bool
    ) -> SarsaState:
        current_q = state.Q[state.prev_state, state.prev_action]
        next_q = state.Q[current_state, action]
        updated_q = current_q + alpha * (reward + gamma * next_q - current_q)
        Q = state.Q.at[state.prev_state, state.prev_action].set(updated_q)

        new_timestep = state.timestep + 1
        new_epsilon = state.epsilon if use_fixed_epsilon else jnp.maximum(0.05, state.epsilon * 0.998)

        return SarsaState(
            Q=Q,
            epsilon=new_epsilon,
            prev_state=current_state,
            prev_action=action,
            prev_reward=reward,
            timestep=new_timestep
        )

    @staticmethod
    def sample(state: SarsaState, key: PRNGKey, current_state: int, use_fixed_epsilon: bool) -> int:
        epsilon = 0.1 if use_fixed_epsilon else state.epsilon

        q_table = state.Q
        q_values = q_table[current_state]
        key, subkey = jax.random.split(key)
        random_action = jax.random.randint(subkey, (), 0, q_values.shape[0])
        greedy_action = jnp.argmax(q_values)

        key, subkey = jax.random.split(key)
        chosen_action = jnp.where(
            jax.random.uniform(subkey) > epsilon,
            greedy_action,
            random_action
        )

        return chosen_action
