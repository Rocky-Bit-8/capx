# Cube Stack Demo Commands

This note records the currently working CaP-X cube-stack commands for this local setup.

## Local Setup

Working directory:

```bash
cd /home/rocky/Code/cap-x-main
```

The project reads the Highland API key from `.env`. Do not commit this file.

```bash
printf 'HIGHLAND_API_KEY=your_key_here\n' > /home/rocky/Code/cap-x-main/.env && chmod 600 /home/rocky/Code/cap-x-main/.env
```

Common environment variables used by the commands:

```bash
LIBERO_CONFIG_PATH=/home/rocky/Code/cap-x-main/.libero MPLCONFIGDIR=/tmp/capx-mpl ROBOT_DESCRIPTIONS_CACHE=/tmp/capx-robot-cache
```

Default LLM endpoint/model in this setup:

```text
https://www.highland-api.top/v1/chat/completions
gemini-3.1-pro-preview:floor
```

## Verified Commands

Privileged oracle smoke test:

```bash
LIBERO_CONFIG_PATH=/home/rocky/Code/cap-x-main/.libero MPLCONFIGDIR=/tmp/capx-mpl ROBOT_DESCRIPTIONS_CACHE=/tmp/capx-robot-cache ./.setup-venv/bin/python -m capx.envs.launch --config-path env_configs/human_oracle_code/robosuite/franka_robosuite_cube_stack_privileged_oracle.yaml --use-oracle-code True --num-workers 1 --total-trials 1 --record-video False --model oracle
```

Standard oracle with local SAM3:

```bash
LIBERO_CONFIG_PATH=/home/rocky/Code/cap-x-main/.libero MPLCONFIGDIR=/tmp/capx-mpl ROBOT_DESCRIPTIONS_CACHE=/tmp/capx-robot-cache ./.setup-venv/bin/python -m capx.envs.launch --config-path env_configs/human_oracle_code/robosuite/franka_robosuite_cube_stack_oracle.yaml --use-oracle-code True --num-workers 1 --total-trials 1 --record-video False --model oracle
```

Single-turn real LLM demo:

```bash
LIBERO_CONFIG_PATH=/home/rocky/Code/cap-x-main/.libero MPLCONFIGDIR=/tmp/capx-mpl ROBOT_DESCRIPTIONS_CACHE=/tmp/capx-robot-cache ./.setup-venv/bin/python -m capx.envs.launch --config-path env_configs/human_oracle_code/robosuite/franka_robosuite_cube_stack_oracle.yaml --use-oracle-code False --num-workers 1 --total-trials 1 --record-video False
```

Multi-turn visual differencing demo:

```bash
LIBERO_CONFIG_PATH=/home/rocky/Code/cap-x-main/.libero MPLCONFIGDIR=/tmp/capx-mpl ROBOT_DESCRIPTIONS_CACHE=/tmp/capx-robot-cache ./.setup-venv/bin/python -m capx.envs.launch --config-path env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm.yaml --num-workers 1 --total-trials 1 --record-video False
```

Interactive Web UI:

```bash
LIBERO_CONFIG_PATH=/home/rocky/Code/cap-x-main/.libero MPLCONFIGDIR=/tmp/capx-mpl ROBOT_DESCRIPTIONS_CACHE=/tmp/capx-robot-cache ./.setup-venv/bin/python -m capx.envs.launch --config-path env_configs/human_oracle_code/robosuite/franka_robosuite_cube_stack_oracle.yaml --web-ui True
```

Open:

```text
http://localhost:8200
```

## Output Directories

Single-turn real LLM:

```text
outputs/gemini-3.1-pro-preview:floor/franka_robosuite_cube_stack_oracle/
```

Multi-turn visual differencing:

```text
outputs/gemini-3.1-pro-preview:floor/franka_robosuite_cube_stack_multiturn_vdm/
```

Generated code is saved under each successful trial directory, for example:

```text
trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/code.py
```

## Local Ports

Web UI:

```text
8200
```

SAM3 server:

```text
8114
```

ContactGraspNet and Pyroki are intentionally not required for the currently stabilized cube-stack path.

```text
8115 ContactGraspNet
8116 Pyroki
```

## Cleanup

Check which process owns a port:

```bash
fuser -v 8200/tcp
```

Stop Web UI:

```bash
fuser -k 8200/tcp
```

Stop SAM3:

```bash
fuser -k 8114/tcp
```

If `robot_descriptions` leaves a stale lock during Pyroki experiments:

```bash
rm /tmp/capx-robot-cache/example-robot-data/.git/shallow.lock
```

## Current Notes

- CUDA/driver is now working on this machine. Current stable cube-stack path uses GPU SAM3 and local checkpoint.
- Local SAM3 checkpoint: `/home/rocky/Code/cap-x-main/sam3_repacked.pt`.
- `LIBERO not installed!` and `R1Pro not installed!` warnings are expected for the cube-stack robosuite demos.
- The verified successful runs returned `Reward: 1.0` and `Task Completed: True`.
