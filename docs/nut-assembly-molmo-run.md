# Nut Assembly Molmo 运行

本配置使用 `FrankaControlNutAssemblyMolmoApi`。它保留 Nut Assembly 原有的 SAM3 分割、深度反投影、几何筛选和 PyRoKi IK，只让 Molmo 为“extruded handle of the brown square nut”提供更准确的二维语义点。Molmo 点会被限制在 SAM3 mask 内；服务不可用、返回空点或点落在图像外时，自动回退到原有的 handle 几何点。因此 Molmo 不会直接生成 3D 位姿，也不会替代 SAM3 或 PyRoKi。

先启动远程 GPU 服务：

```bash
ssh -p 19154  root@connect.weste.seetacloud.com \
  '/root/autodl-tmp/start_molmo_vllm.sh'
```

另开终端保持隧道：

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 8122:127.0.0.1:8122 \
  -p 19154  root@connect.weste.seetacloud.com
```

确认服务已经完成模型加载后检查：

```bash
curl -fsS http://127.0.0.1:8122/v1/models
```

## RTX 4080 显存配置

如果日志出现 `No available memory for the cache blocks`，说明模型权重和 CUDA
运行开销后没有足够 KV cache。编辑远程 `/root/autodl-tmp/start_molmo_vllm.sh`，将
vLLM 参数调整为：

```bash
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 2048 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 1 \
  --enforce-eager \
  --trust-remote-code \
```

`max-model-len` 和 `max-num-batched-tokens` 同时限制为 2048，`max-num-seqs=1`
限制并发，`enforce-eager` 避免 CUDA graph 额外占用显存。若仍无 KV cache，将前两项
长度同时降至 1024。不要使用原来的 `--gpu-memory-utilization 0.45`，4080 上可能
无法为 KV cache 留出空间。

修改后重启远程服务：

```bash
ssh -p 12469 root@connect.weste.seetacloud.com \\
  'if test -f /root/autodl-tmp/molmo-vllm.pid; then kill "$(cat /root/autodl-tmp/molmo-vllm.pid)" 2>/dev/null || true; fi; /root/autodl-tmp/start_molmo_vllm.sh'
```

等待日志出现 `Application startup complete` 后，再建立 SSH 隧道。服务启动脚本
使用 CUDA/PyTorch 检测，不要求设备节点必须叫 `/dev/nvidia0`；容器中出现
`/dev/nvidia2` 也属于正常映射。

## 云服务器地址变更

更换云服务器或实例重启后，通常只需要更新 SSH 连接信息，不需要修改 Nut Assembly 配置或公共 API。先从云平台复制新的登录命令，例如：

```bash
ssh -p <SSH_PORT> <SSH_USER>@<SSH_HOST>
```

然后把本文前两条 SSH 命令中的以下三项一起替换：

- `<SSH_HOST>`：云平台提供的新域名或公网 IP，例如 `connect.weste.seetacloud.com`。
- `<SSH_PORT>`：新的 SSH 端口，例如 `12469`。
- `<SSH_USER>`：远程用户名，当前示例为 `root`。

更新后的启动命令为：

```bash
ssh -p <SSH_PORT> <SSH_USER>@<SSH_HOST> \
  '/root/autodl-tmp/start_molmo_vllm.sh'
```

更新后的隧道命令为：

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 8122:127.0.0.1:8122 \
  -p <SSH_PORT> <SSH_USER>@<SSH_HOST>
```

只要本地转发端口仍为 `8122`，程序访问地址继续使用 `http://127.0.0.1:8122`，无需随云服务器公网地址变化。如果本地 `8122` 已被占用，可以把隧道参数左侧改为其他端口，例如 `-L 18122:127.0.0.1:8122`；此时还需将 Molmo 配置中的本地服务地址同步改为 `http://127.0.0.1:18122`。右侧的远程端口 `8122` 只有在远程服务实际改用其他端口时才需要修改。

地址更新后，建议按以下顺序验证：先用新的 SSH 信息正常登录；在远程机器上执行 `curl -fsS http://127.0.0.1:8122/v1/models`；退出后重新建立隧道；最后在本机执行同一条 `curl`。

在项目根目录运行一次测试：

```bash
LIBERO_CONFIG_PATH=/home/rocky/Code/cap-x-main/.libero \
MPLCONFIGDIR=/tmp/capx-mpl \
ROBOT_DESCRIPTIONS_CACHE=/tmp/capx-robot-cache \
./.setup-venv/bin/python -m capx.envs.launch \
  --config-path /home/rocky/Code/cap-x-main/env_configs/nut_assembly/franka_robosuite_nut_assembly_multiturn_molmo.yaml \
  --num-workers 1 --total-trials 1 --record-video True
```

服务启动脚本立即返回并不代表端口已就绪；应等待日志出现 `Application startup complete`。若隧道提示 `Connection refused`，先检查远程 `curl`，再重新建立隧道。该变体不修改原始 Nut Assembly 配置或视觉 API。
