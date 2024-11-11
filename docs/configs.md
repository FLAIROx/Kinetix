# Configuration

- [Configuration](#configuration)
  - [Configuration Headings](#configuration-headings)
    - [Env](#env)
    - [Env Size](#env-size)
    - [Learning](#learning)
    - [Misc](#misc)
    - [Eval](#eval)
    - [Eval Env Size](#eval-env-size)
    - [Train Levels](#train-levels)
    - [Model](#model)
    - [UED](#ued)


We use [hydra](hydra.cc) for all of our configurations, and we use [hierarchical configuration](https://hydra.cc/docs/tutorials/structured_config/schema/) to organise everything better. 

In particular, we have the following configuration headings, with the base `ppo` config looking like:
```yaml
defaults:
  - env: entity
  - env_size: s
  - learning:
    - ppo-base
    - ppo-rnn
  - misc: misc
  - eval: s
  - eval_env_size: s
  - train_levels: random
  - model:
    - model-base
    - model-transformer
  - _self_
seed: 0
```

## Configuration Headings
### Env
This controls the environment to be used.
#### Preset Options
We provide two options in `configs/env`, namely `entity` and `symbolic`; each of these can be used by running `python3 experiments/ppo.py env=symbolic` or `python3 experiments/ppo.py env=entity`. If you wish to customise the options further, you can add any of the following subkeys (e.g. by running `python3 experiments/ppo.py env=symbolic env.dense_reward_scale=0.0`):
#### Individual Subkeys
- `env.env_name`: The name of the environment, with controls the observation and action space.
- `env.dense_reward_scale`: How large the dense reward scale is, set this to zero to disable dense rewards.
- `env.frame_skip`: The number of frames to skip, setting this to 2 (the default) seems to perform better.
### Env Size
This controls the maximum number of shapes present in the simulation. This has two important tradeoffs, namely speed and representational power: Small environments run much faster but some complex environments require a large number of shapes. See `configs/env_size`
#### Preset Options
- `s`: The `small` preset
- `m`: `Medium` preset
- `l`: `Large` preset
- `custom`: Allows the use of a custom environment size loaded from a json file (see [here](#train-levels) for more).
#### Individual Subkeys
- `num_polygons`: How many polygons
- `num_circles`: How many circles
- `num_joints`: How many joints
- `num_thrusters`: How many thrusters
- `env_size_name`: "s", "m" or "l"
- `num_motor_bindings`: How many different joint bindings are there, meaning how many different actions are there associated with joints. All joints with the same binding will have the same action applied to them.
- `num_thruster_bindings`: How many different thruster bindings are there
- `env_size_type`: "predefined" or "custom"
- `custom_path`: **Only for env_size_type=custom**, controls the json file to load the custom environment size from.
### Learning
This controls the agent's learning, see `configs/learning`
#### Preset Options
- `ppo-base`: This has all of the base PPO parameters, and is used by all methods
- `ppo-rnn`: This has the PureJaxRL settings for some of PPO's hyperparameters (mainly `num_steps` is different)
- `ppo-sfl`: This has the SFL-specific value of `num_steps`
- `ppo-ued`: This has the JAXUED-specific `num_steps` and `outer_rollout_steps`
#### Individual Subkeys
- `lr`: Learning Rate
- `anneal_lr`: Whether to anneal LR
- `warmup_lr`:  Whether to warmup LR
- `peak_lr`: If warming up, the peak
- `initial_lr`: If warming up, the initial LR
- `warmup_frac`: If warming up, the warmup fraction of training time
- `max_grad_norm`: Maximum grad norm
- `total_timesteps`: How many total environment interactions must be run
- `num_train_envs`: Number of parallel environments to run simultaneously
- `num_minibatches`: Minibatches for PPO learning
- `gamma`:  Discount factor
- `update_epochs`: PPO update epochs
- `clip_eps`: PPO clipping epsilon
- `gae_lambda`: PPO Lambda for GAE
- `ent_coef`: Entropy loss coefficient
- `vf_coef`: Value function loss coefficient
- `permute_state_during_training`: If true, the state is permuted on every reset.
- `filter_levels`: If true, and we are training on random levels, this filters out levels that can be solved by a no-op
- `level_filter_n_steps`: How many steps to allocate to the no-op policy for filtering
- `level_filter_sample_ratio`: How many more levels to sample than required (ideally `level_filter_sample_ratio` is more than the fraction that will be filtered out).
- `num_steps`: PPO rollout length
- `outer_rollout_steps`: How many learning steps to do for e.g. PLR for each rollout (see the [Craftax paper](https://arxiv.org/abs/2402.16801) for a more in-depth explanation).
### Misc
There are a plethora of miscellaneous options that are grouped under the `misc` category. There is only one preset option, `configs/misc/misc.yaml`.
#### Individual Subkeys
- `group`: Wandb group ("auto" usually works well)
- `group_auto_prefix`: If using group=auto, this is a user-defined prefix
- `save_path`: Where to save checkpoints to
- `use_wandb`: Should wandb be logged to
- `save_policy`: Should we save the policy
- `wandb_project`: Wandb project
- `wandb_entity`: Wandb entity, leave as `null` to use your default one
- `wandb_mode` : Wandb mode
- `video_frequency`: How often to log videos (they are quite large)
- `load_from_checkpoint`: WWandb artifact path to load from
- `load_only_params`: Whether to load just the network parameters or entire train state.
- `checkpoint_save_freq`: How often to log checkpoits
- `checkpoint_human_numbers`: Should the checkpoints have human-readable timestep numbers
- `load_legacy_checkpoint`: Do not use
- `load_train_levels_legacy`: Do not use
- `economical_saving`: If true, only saves a few important checkpoints for space conservation purposes.
### Eval
This option (see `configs/eval`) controls how evaluation works, and what levels are used.
#### Preset Options
- `s`: Eval on the `s` hand-designed levels located in `worlds/s`
- `m`: Eval on the `m` hand-designed levels located in `worlds/m`
- `l`: Eval on the `l` hand-designed levels located in `worlds/l`
- `eval_all`: Eval on all of the hand-designed eval levels
- `eval_auto`: If `train_levels` is not random, evaluate on the training levels.
- `mujoco`: Eval on the recreations of the mujoco tasks.
- `eval_general`: General option if you are planning on overwriting most options.
#### Individual Subkeys
- `eval_levels`: List of eval levels or the string "auto"
- `eval_num_attempts`: How many times to eval on the same level
- `eval_freq`: How often to evaluate
- `EVAL_ON_SAMPLED`: If true, in `plr.py` and `sfl.py`, evaluates on a fixed set of randomly-generated levels

### Eval Env Size
This controls the size of the evaluation environment. This is crucial to match up with the size of the evaluation levels.
#### Preset Options
- `s`: Same as the `env_size` option.
- `m`: Same as the `env_size` option.
- `l`: Same as the `env_size` option.
### Train Levels
Which levels to train on.
#### Preset Options
- `s`: All of the `s` holdout levels
- `m`: All of the `m` holdout levels
- `l`: All of the `l` holdout levels
- `train_all`: All of the levels from all 3 holdout sets
- `mujoco`: All of the mujoco recreation levels.
- `random`: Train on random levels
#### Individual Subkeys
- `train_level_mode`: "random" or "list"
- `train_level_distribution`: if train_level_mode=random, this controls which distribution to use. By default `distribution_v3`
- `train_levels_list`: This is a list of levels to train on.
### Model
This controls the model architecture and options associated with that.
#### Preset Options
We use both of the following:
- `model-base`
- `model-entity`
#### Individual Subkeys
`fc_layer_depth`: How many layers in the FC model
`fc_layer_width`: How wide is each FC layer
`activation`: NN activation
`recurrent_model`: Whether or not to use recurrence
The following are just relevant when using `env=entity`
`transformer_depth`: How many transformer layers to use
`transformer_size`: How large are the KQV vectors
`transformer_encoder_size`: How large are the initial embeddings
`num_heads`: How many heads, must be a multiple of 4 and divide `transformer_size` evenly.
`full_attention_mask`: If true, all heads use the full attention mask
`aggregate_mode`: `dummy_and_mean` works well.
### UED
Options pertaining to UED (i.e., when using the scripts `plr.py` or `sfl.py`)
#### Preset Options
- `sfl`
- `plr`
- `accel`
#### Individual Subkeys
See the individual files for the configuration options used.
For SFL, we have:

- `sampled_envs_ratio`: How many environments are from the SFL buffer and how many are randomly generated
- `batch_size`: How many levels to evaluate learnability on per batch
- `num_batches`: How many batches to run when choosing the most learnable levels
- `rollout_steps`: How many steps to rollout for when doing the learnability calculation.
- `num_to_save`: How many levels to save in the learnability buffer