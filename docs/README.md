# Documentation
This is intended to provide some more details about how Kinetix works, including more in-depth examples. If you are interested in the configuration options, see [here](./configs.md).

- [Documentation](#documentation)
  - [Different Versions of Kinetix Environments](#different-versions-of-kinetix-environments)
    - [Action Spaces](#action-spaces)
    - [Observation Spaces](#observation-spaces)
  - [Resetting Functionality](#resetting-functionality)
    - [🔪 Sharp Bits 🔪](#-sharp-bits-)
  - [Evaluate an Agent](#evaluate-an-agent)
  - [Using Kinetix to easily design your own JAX Environments](#using-kinetix-to-easily-design-your-own-jax-environments)
    - [Step 1 - Design an Environment](#step-1---design-an-environment)
    - [Step 2 - Export It](#step-2---export-it)
    - [Step 3 - Import It](#step-3---import-it)
    - [Step 4 - Train](#step-4---train)


## Different Versions of Kinetix Environments
We provide several different variations on the standard Kinetix environment, where the primary difference is the action and observation spaces.
Each of these can be made using the `make_kinetix_env` function, passing different values of `action_type` and `observation_type`
### Action Spaces
For all action spaces, the agent can control joints and thrusters. Joints have a property `motor_binding`, which is a way to tie different joints to the same action. Two joints that have the same binding will always perform the same action, likewise for thrusters.

We have three observation spaces, discrete, continuous and multi-discrete (which is the default). 
- **Discrete** has `2 * num_motor_bindings + num_thruster_bindings + 1` options, one of which can be active at any time. There are two options for every joint, i.e., backward and forward at full power. There is one option for each thruster, to activate it at full power. The final option is a no-op, meaning that no torque or force is applied to joints/thrusters.
- **Continuous** has shape `num_motor_bindings + num_thruster_bindings`, where each motor element can take a value between -1 and 1, and thruster elements can take values between 0 and 1.
- **Multi-Discrete**: This is a discrete action space, but allows multiple joints and thrusters to be active at any one time. The agent must output a flat vector of size `3 * num_motor_bindings + 2 * num_thruster_bindings`. For joints, each group of three represents a categorical distribution of `[0, -1, +1]` and for thrusters it represents `[0, +1]`.

### Observation Spaces
We provide three primary observation spaces, Symbolic-Flat (called just symbolic), Symbolic-Entity (called entity, which is also the default) and Pixels.
- **Symbolic-Flat** returns a large vector, which is the flattened representation of all shapes and their properties.
- **Symbolic-Entity** also returns a vector representation of all entities, but does not flatten it, instead returning it in a form that can be used with permutation-invariant network architectures, such as transformers. By default this uses a permutation invariant MLP architecture, but you can use the transformer-based model we used in our paper by setting `model.permutation_invariant_mlp=False` in the command line config.
- **Pixels** returns an image representation of the scene. This is partially observable, as features such as the restitution and density of shapes is not shown.


Each observation space has its own pros and cons. **Symbolic-Flat** is the fastest by far, but has two clear downsides. First, it is restricted to a single environment size, e.g. a model trained on `small` cannot be run on `medium` levels. Second, due to the large number of symmetries (e.g. any permutation of the same shapes would represent the same scene but would look very different in this observation space), this generalises worse than *entity*.

**Symbolic-Entity** is faster than pixels, but slower than Symbolic-Flat (although not significantly slower if we use the permutation invariant MLP model instead of the transformer). However, it can be applied to any number of shapes, and is natively permutation invariant. For these reasons we chose it as the default option.

Finally, **Pixels** runs the slowest, and also requires more memory, which means that we cannot run as many parallel environments. However, pixels is potentially the most general format, and could theoretically allow transfer to other domains and simulators.


## Resetting Functionality
We have two primary resetting functions that control the environment's behaviour when an episode ends. The first of these is to train on a known, predefined set of levels, and resetting samples a new level from this set. In the extreme case, this also allows training only on a single level in the standard RL manner. The other main way of resetting is to sample a *random* level from some distribution, meaning that it is exceedingly unlikely to sample the same level twice.

### 🔪 Sharp Bits 🔪
When using `make_kinetix_env`, you can specify the reset function by passing `reset_func`, a callable, taking in an rng and returning an `EnvState`.

This reset function can be made manually, or using the `make_reset_func` function.
Either `reset_mode=ResetMode.RANDOM` can be passed, in which case random levels will be generated upon reset, or if `reset_mode=ResetMode.LIST`, then the argument `train_levels_list` must be specified as a list of strings, containing levels to load. In this case, a random level will always be chosen from this list upon reset.

All of these result in an environment object that has gymnax auto-reset behaviour, i.e., when you run `.step`, and the episode terminates, it automatically resets to a new initial state.

However, you can also manually control which level the environment resets to. This can be done using the `.reset` and `.step` functions, passing in `override_reset_state` as the final argument. This is optional and, when provided, will override any behaviour specified by `reset_func`.
For instance, 
```python
obs, env_state = env.reset(_rng, env_params) # resets to a level given by `env.reset_func`
obs, env_state = env.reset(_rng, env_params, level) # resets to `level`
```

```python
obs, env_state, reward, done, info = env.step(_rng, env_state, action, env_params) # if the episode terminates, resets to `reset_func`'s level. 
obs, env_state, reward, done, info = env.step(_rng, env_state, action, env_params, level) # if the episode terminates, it would reset to `level`, otherwise it would just step normally.
```

Finally, if you use `make_kinetix_env(..., reset_func=None)`, then the environment will throw an exception whenever you try and `step` or `reset` without explicitly passing a `override_reset_state`.

## Evaluate an Agent
If you train an agent using one of the scripts in `experiments`, a checkpoint will be saved by default to wandb. Then, this checkpoint can be loaded. See `examples/example_inference.py` for how to perform inference. This script can be used as follows:

```bash
python examples/example_inference.py misc.load_from_checkpoint=<WANDB_ARTIFACT_PATH>
```

You can also see the agent play in the editor as follows, and then pressing play.
```bash
python kinetix/editor.py agent_taking_actions=true misc.load_from_checkpoint=<WANDB_ARTIFACT_PATH>
```


> [!WARNING]
> Ensure that the observation space and action space are the same between inference and training (e.g. if during training you had `env.action_type=multi_discrete` then the same option must be present when running inference).

## Using Kinetix to easily design your own JAX Environments
Since Kinetix has a general physics engine, you can design your own environments and train RL agents on them very fast! This section in the docs describes this pipeline.
### Step 1 - Design an Environment
You can go to our [online editor](https://kinetix-env.github.io/gallery.html?editor=true) and have a look at the [gallery](https://kinetix-env.github.io/gallery.html) if you need some inspiration.

The following two images show the main editor page, and then the level I designed, where you have to spin the ball the right way. While designing the level, you can play it to test it out, seeing if it is possible and of the appropriate difficulty.


<p align="middle">
  <img src="../images/docs/edit-1.png" width="49%" />
  <img src="../images/docs/edit-2.png" width="49%" />
</p>

### Step 2 - Export It
Once you are satisfied with your level, you can download it as a json file by using the button on the bottom left. Once this is downloaded, move it to `/path/to/repo/worlds/custom/my_custom_level.json`.


### Step 3 - Import It
In python, you can import the level as follows, see `examples/example_premade_level_replay.py` for an example.
```python
from kinetix.util.saving import load_from_json_file
level, static_env_params, env_params = load_from_json_file("/path/to/repo/worlds/custom/my_custom_level.json")
```

### Step 4 - Train
You can use the above if you want to import the level and play around with it. If you want to train an RL agent on this level, you can do the following (see [here](https://github.com/FLAIROx/Kinetix?tab=readme-ov-file#training-on-a-single-hand-designed-level) from the main README).

```commandline
python3 experiments/ppo.py env_size=custom \
                           env_size.custom_path=/path/to/repo/worlds/custom/my_custom_level.json \
                           train_levels=s \
                           train_levels.train_levels_list='["/path/to/repo/worlds/custom/my_custom_level.json"]' \
                           eval=eval_auto
```

And the agent will start training, with videos like this on [wandb](https://wandb.ai).
<p align="middle">
  <img src="../images/docs/wandb.gif" width="49%" />
</p>