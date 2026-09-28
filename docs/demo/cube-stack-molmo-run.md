# Cube Stack Molmo 运行说明

先在远程 GPU 服务器启动 Molmo 服务：

```bash
ssh -p 47322 root@connect.westb.seetacloud.com \
  '/root/autodl-tmp/start_molmo_vllm.sh'
```

另开本地终端建立端口转发，使 CaP-X 的 `127.0.0.1:8122` 映射到远程 Molmo：

```bash
ssh -N -L 8122:127.0.0.1:8122 \
  -p 47322 root@connect.westb.seetacloud.com
```

保持该终端运行，并在第三个终端检查服务：

```bash
curl -fsS http://127.0.0.1:8122/v1/models
```

确认返回模型 `allenai/Molmo2-8B` 后，在项目根目录运行一次 cube-stack：

```bash
LIBERO_CONFIG_PATH=/home/rocky/Code/cap-x-main/.libero \
MPLCONFIGDIR=/tmp/capx-mpl \
ROBOT_DESCRIPTIONS_CACHE=/tmp/capx-robot-cache \
./.setup-venv/bin/python -m capx.envs.launch \
  --config-path /home/rocky/Code/cap-x-main/env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_molmo.yaml \
  --num-workers 1 \
  --total-trials 1 \
  --record-video True
```

该配置使用 `FrankaControlMolmoApi`，Molmo 只负责点提示；SAM3、Contact-GraspNet、PyRoKi 仍由本地配置启动。若服务连接失败，先确认 SSH 隧道未退出，再重新执行健康检查。
