# Nut Assembly 仿真渲染与三维视图接口

运行该 Nut Assembly 配置并加 `--web-ui True` 后，`capx/web/async_trial_runner.py` 会自动打开 `enable_render`、`viser_debug`，通过 Viser 代理显示场景。

核心实现是 `capx/envs/simulators/robosuite_nut_assembly.py`：Robosuite 使用 `birdview`（512×512）作为俯视渲染相机，观测统一放在 `robot0_robotview` 下。`get_observation()` 返回 RGB、真实深度、`intrinsics`、`pose/pose_mat`，可供 SAM3 和深度反投影使用。

`render()` 获取 birdview 帧，`render_wrist()` 获取末端相机帧。`_update_viser_server()` 通过 `depth_color_to_pointcloud()` 生成点云，并显示相机视锥、机器人 URDF、nut/peg 与抓取坐标系；3D 数据只更新内存中的 Viser 场景，不写 PNG。
