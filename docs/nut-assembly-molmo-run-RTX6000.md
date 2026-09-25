# Nut Assembly + Molmo（RTX 6000D）运行链路

本文对应当前 RTX 6000D 云实例：

```text
SSH host: connect.weste.seetacloud.com
SSH port: 19154
GPU: NVIDIA RTX 6000D（sm120，约 89 GiB）
远程 Molmo API: 127.0.0.1:8122
```

Nut Assembly 的 3D 链路仍由 SAM3、深度反投影和 PyRoKi 完成；Molmo 只为
`extruded handle of the brown square nut` 提供二维语义点，并限制在 SAM3 mask
内，失败时自动回退到原几何点。

## 1. 启动远程 Molmo

远程启动脚本 `/root/autodl-tmp/start_molmo_vllm.sh` 应包含以下关键设置：

```bash
export PATH="/root/autodl-tmp/molmo-venv/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
```

RTX 6000D 使用 sm120；当前 vLLM/FlashInfer 组合会在 sampler JIT 阶段误报
`FlashInfer requires GPUs with sm75 or higher`，因此必须关闭 FlashInfer sampler。
显存充足时使用：

```text
--dtype bfloat16
--gpu-memory-utilization 0.80
--max-model-len 4096
--max-num-batched-tokens 4096
--trust-remote-code
```

登录并启动（脚本会后台运行）：

```bash
ssh -p 19154 root@connect.weste.seetacloud.com \
  '/root/autodl-tmp/start_molmo_vllm.sh'
```

远程检查模型加载完成：

```bash
ssh -p 19154 root@connect.weste.seetacloud.com \
  'curl -fsS http://127.0.0.1:8122/v1/models'
```

输出中应包含 `"id":"allenai/Molmo2-8B"`。首次启动需要等待模型编译和 CUDA
graph 捕获，通常约 1 分钟；不要只根据启动脚本立即返回判断服务已就绪。

若需重启：

```bash
ssh -p 19154 root@connect.weste.seetacloud.com '
if test -f /root/autodl-tmp/molmo-vllm.pid; then
  kill "$(cat /root/autodl-tmp/molmo-vllm.pid)" 2>/dev/null || true
fi
rm -f /root/autodl-tmp/molmo-vllm.log
/root/autodl-tmp/start_molmo_vllm.sh
'
```

日志位置：`/root/autodl-tmp/molmo-vllm.log`。

## 2. 建立本地 SSH 隧道

保持以下命令运行：

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 8122:127.0.0.1:8122 \
  -p 19154 root@connect.weste.seetacloud.com
```

另开终端验证本地转发：

```bash
curl -fsS http://127.0.0.1:8122/v1/models
```

本地 `8122` 被占用时，将左侧改为例如 `18122`：
`-L 18122:127.0.0.1:8122`，并把 `capx/integrations/vision/molmo.py` 中的
`SERVICE_URL` 改为 `http://127.0.0.1:18122/v1`。

## 3. 运行 Nut Assembly

在项目根目录执行：

```bash
LIBERO_CONFIG_PATH=/home/rocky/Code/cap-x-main/.libero \
MPLCONFIGDIR=/tmp/capx-mpl \
ROBOT_DESCRIPTIONS_CACHE=/tmp/capx-robot-cache \
MUJOCO_GL=egl \
./.setup-venv/bin/python -m capx.envs.launch \
  --config-path /home/rocky/Code/cap-x-main/env_configs/nut_assembly/franka_robosuite_nut_assembly_multiturn_molmo.yaml \
  --num-workers 1 --total-trials 1 --record-video True
```

该配置启动本地 SAM3（8114）和 PyRoKi（8116），并通过隧道调用远程 Molmo
（8122）。首次运行建议使用 `--num-workers 1 --total-trials 1` 做冒烟测试，确认
链路正常后再提高并发和试验数。

## 4. 常见问题

- `No available memory for the cache blocks`：确认 RTX 6000D 使用
  `--gpu-memory-utilization 0.80`；必要时将长度参数同时降到 2048。不要把
  `gpu-memory-utilization` 降到 0.45。
- `FlashInfer requires GPUs with sm75 or higher`：确认启动脚本中存在
  `export VLLM_USE_FLASHINFER_SAMPLER=0`，然后删除 pid/log 并重启。
- 隧道 `Connection refused`：先执行远程 `curl`，等待日志出现
  `Application startup complete` 后再重建隧道。
- 云实例更换后：只需替换本文命令中的 host/port/user；远程服务端口仍为 8122，
  本地 CaP-X 配置无需改动。
