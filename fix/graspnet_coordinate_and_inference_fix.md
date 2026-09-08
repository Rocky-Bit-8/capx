# GraspNet 坐标与推理修复说明

## 本次修复

### 1. 推理模式

`capx/serving/launch_contact_graspnet_server.py` 在加载 `model.pt` 后现在显式调用
`model.eval()`。此前模型保持 PyTorch 默认的 train 模式，导致 Dropout 和
BatchNorm 在服务推理时仍然生效，抓取置信度不稳定，经过过滤后经常没有候选。

`contact_grasp_estimator.py` 的模型前向也放入 `torch.inference_mode()`，避免推理
保留 autograd 图并降低不必要的显存占用。

诊断中同一输入的结果为：train 模式只有约 `3/2048` 个点超过阈值，eval 模式约
`123/2048` 个点超过阈值。这是 GraspNet 空抓取的主要原因。

### 2. viser 抓取框坐标

`grasp_sample` 是 Contact-GraspNet 输出的相机坐标系抓取矩阵，而 viser 场景使用
机器人基座坐标系。此前代码直接绘制 `grasp_sample`，所以调试画面中的抓取框会偏离
真实物体。

现在使用：

```text
T_base_grasp = T_base_camera @ T_camera_grasp
```

并在绘制前应用 `obs["robot0_robotview"]["pose_mat"]`。这只修正可视化坐标，不改变
实际发送给机器人/仿真的抓取目标。

## 已确认的坐标约定

- RGB、深度和内参属于同一个 `robot0_robotview` 相机。
- 深度反投影使用 OpenCV 相机系：`x` 向右、`y` 向下、`z` 指向相机前方。
- `convert_cam_coords=True` 会在 Contact-GraspNet 内部转换到训练坐标，输出时再转换
  回 OpenCV 相机系；服务端随后用 `pose_mat @ grasp_cam` 转到基座系。
- 抓取矩阵的 `z` 列是 approach 方向，`x` 列是夹爪闭合方向。
- `center_to_tip` 默认没有额外偏移。项目侧的 `+0.12 m` 是现有抓取 frame 到
  gripper/TCP 的固定偏移，不是 GraspNet 自动估计出来的夹爪长度。

## 数值验证结论

- GT 点投影回图像与渲染物体位置一致。
- depth -> OpenCV camera -> base 的点云回环通过；peg 高度误差约为毫米以下，nut
  平面位置误差小于 `1 mm`。
- 因此当前主要故障不是 RGB/depth/K/`pose_mat` 的基本坐标链路，而是服务端此前以
  train 模式执行模型。

## 独立问题：EE/API frame

为恢复 Cube Stack 与原版的一致行为，公共 `pyroki.py` 已移除额外的
`Ry(pi) @ Rz(pi)` EE 姿态补偿。当前本地 IK 按原版方式直接解释 API 传入的目标姿态，
再结合 base pose 转换到 MuJoCo 坐标。

`Rz(pi/2)`、`Rx(pi)` 等历史约定仍存在于其他观测或专用控制路径中；它们属于后续
需要单独验证的 EE/API frame 一致性问题，不应与 GraspNet 相机坐标问题混为一谈。
本次没有回退其他文件，避免影响 Nut Assembly 之外的公共行为。

## 重新验证

重启旧的 Contact-GraspNet 服务后再测试，避免旧进程继续使用未修复的 train 模式。重点
观察服务日志中的 `model set to eval mode`，以及严格过滤后是否仍然返回空抓取。

## Nut Assembly 专属修复

后续对原始场景的复查表明，Cube Stack 的仿真抓取会直接返回物体中心和固定顶抓姿态，
并未走与 Nut Assembly 相同的 Contact-GraspNet 推理分支，因此不能用它证明小物体分支
正常。Nut Assembly 的实际失败还有一个任务特有原因：SAM3 对
`extruded handle of the brown square nut` 的最高分 mask 经常覆盖整颗带孔螺母，而不是
只覆盖狭窄把手。未过滤候选会落到桌面或螺母主体，`filter_grasps=True` 后则经常为空。

修复限定在 `FrankaControlNutAssemblyVisualApi`：检测到带明显内孔的完整螺母 mask 时，
利用内孔与轮廓的相对位置取出窄端把手区域，再把该区域交给原有 GraspNet 服务。对于
薄把手仍被严格过滤掉的情况，使用把手 mask 的有效深度点生成固定向下的保守抓取姿态。
这个 fallback 只在 Nut Assembly API 内生效，没有改变公共 GraspNet 服务、阈值或其他
任务的抓取后处理。

## Nut Assembly IK 修复

视觉 Nut Assembly 配置此前调用 `init_curobo()` 的默认地址 `127.0.0.1:8117`，但
`franka_robosuite_nut_assembly.yaml`、普通 `multiturn` 和 `multiturn_vf` 配置没有
启动 cuRobo server，因此请求必然得到 connection refused。三份配置现在都显式启动
`launch_curobo_server.main`；使用 `FrankaControlApiReduced` 的配置保持原有 PyRoKi
路径不变。

另外，cuRobo 的末端是 `panda_hand`，而 Robosuite Panda 控制/观测的
`gripper0_right_eef` 相对它有固定的 `Rz(-90 deg)` 与本地 `+0.097 m` 变换。Nut
Assembly visual API 现在在每次 IK 请求前将 API 夹爪中心位姿转换到 `panda_hand`，并
继续用 API 夹爪中心位姿做实际位姿验收。该变换在加载的 Robosuite MJCF 上测得，平移
误差小于 `5e-16 m`、旋转误差小于 `2e-14 deg`。

cuRobo server 在 IK 失败时会返回 `success: false`。client 现在检查该字段并立即抛出
错误，不再把 solver 的 best-effort 数值当作可执行关节解。Nut visual API 捕获该
明确失败后，会调用 Nut 专属的本地 MuJoCo 多初值 IK 做可达性复核：若本地能收敛，
说明是 cuRobo 求解器/初始化问题，并使用该任务专属候选继续执行；若本地也无法收敛，
才报告目标确实不可达。该逻辑不会改变公共 PyRoKi、公共 GraspNet 或其他任务路径。

例如目标
`[-0.7071, 0, 0, 0.7071, 0.3911, 0.2215, -0.1390]` 是
`goto_pose(..., z_approach=0.05)` 生成的 approach 阶段 cuRobo 目标，并非 GraspNet
原始输出。逆变换后对应 API 夹爪中心位置 `[0.3911, 0.2215, -0.0420]` 与顶抓姿态
`[0, 0, 1, 0]`；在当前 Robosuite Panda 模型上本地多初值 IK 可零误差收敛，因此该
案例属于 cuRobo 求解失败而非目标配件生成错误。

本机验证：Nut Assembly 专项测试 `5 passed`，Robosuite 固定末端变换测量通过，代码
编译通过。当前环境无可用 NVIDIA 驱动，未能进行 GPU cuRobo server 的端到端启动测试。
