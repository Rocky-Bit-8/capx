# GraspNet 模块说明

本项目使用 Contact-GraspNet 从 RGB-D 深度图和相机内参生成六自由度抓取候选。主要模块如下。

## 服务入口

`capx/serving/launch_contact_graspnet_server.py` 是独立的 FastAPI 服务，默认监听 8115 端口。启动时加载第三方 Contact-GraspNet 模型和 checkpoint，提供 `/plan`、`/plan_point_clouds` 与 `/plan_evenly` 接口。`/plan` 接收 base64 编码的 depth、`cam_K` 和 `segmap`，先提取完整点云及目标分段点云，再调用模型推理。服务还支持局部区域、候选过滤、前向次数和随机视角重试。

## 调用封装

`capx/integrations/vision/graspnet.py` 是 Python 客户端封装。`init_contact_graspnet()` 返回基于 HTTP 的规划函数，负责编码 NumPy 数组、提交请求并解码结果；`init_contact_graspnet_point_clouds()` 用于已经准备好的点云。常规控制模块通过这个封装访问服务，避免在控制进程内重复加载 GPU 模型。

## 点云与几何工具

`capx/utils/graspnet_utils.py` 提供相机朝向、半球采样和随机视角等工具。服务中的 `extract_point_clouds()` 将深度图按相机内参转换为米制相机坐标点云，并依据 `segmap_id` 提取目标区域。随机重试会在临时相机坐标中再次预测，再尝试变换回输入坐标系。

## 第三方模型

模型实现位于 `capx/third_party/contact_graspnet_pytorch/`，checkpoint 位于其 `checkpoints/contact_graspnet/checkpoints/`。核心类是 `GraspEstimator`，主要方法为 `extract_point_clouds()` 和 `predict_scene_grasps()`。

## 输出格式

返回结果包含抓取矩阵、分数和接触点。抓取矩阵通常是相机坐标系下的 `N x 4 x 4` 齐次变换：左上 `3 x 3` 是夹爪方向，最后一列前三项是夹爪参考位置；接触点是 `N x 3` 三维坐标，单位为米。当前服务没有返回坐标系元数据，也不强制接触点投影到 2D mask 内，因此调用方应自行进行 mask、深度、碰撞和机器人基座坐标校验。

`sam_test/verify_nut_assembly.py` 是验证脚本，保存原始候选、Top20、NumPy 数组以及 RGB 投影可视化。默认保留未过滤候选；需要严格后处理时使用 `--filter-grasps`。

## max_retries 设置

重试参数在三处出现。验证脚本的命令行参数位于 `sam_test/verify_nut_assembly.py` 的 `--max-retries`，当前默认值为 5；它会原样写入 `/plan` 请求的 `max_retries` 字段。例如 `--max-retries 100` 会允许服务最多进行 100 次随机相机视角重试。HTTP 客户端 `capx/integrations/vision/graspnet.py` 的两个规划函数默认值是 1，服务模型 `PlanRequest` 和 `PlanPointCloudsRequest` 的默认值是 7（位于 `capx/serving/launch_contact_graspnet_server.py`）。实际请求中传入的字段优先于服务端默认值。

服务在第一次点云推理没有目标抓取时进入重试循环，每次改变临时视角并重新预测；一旦获得候选通常应停止。服务返回逻辑已经修正为始终返回初始推理或最后一次重试的结果，因此 `max_retries=0` 表示只执行初始推理，不会被错误清空。该参数表示重试次数，不是模型采样数量；增大它不能保证候选质量，只会增加计算时间和随机视角结果的不确定性。

## `/plan` 关键参数

验证脚本当前发送的参数含义如下：

- `local_regions`：是否只在 `segmap_id` 对应的目标局部点云上推理。默认值为 `true`（除非使用 `--global-regions`）；关闭后使用完整场景点云，可能包含更多背景和干扰物。
- `filter_grasps`：是否启用 Contact-GraspNet 的候选后处理过滤。当前脚本默认是 `false`，会保留全部原始候选；加上 `--filter-grasps` 后才过滤碰撞、姿态和几何条件不满足的候选。
- `skip_border_objects`：是否跳过触及深度图边界的物体。当前固定为 `false`，所以边界附近的点不会因为这个条件被预先排除。
- `z_range`：从深度图恢复点云时允许的相机坐标系深度范围，单位为米。当前 `[0.2, 2.0]` 表示只使用距离相机 0.2 到 2.0 米的点。
- `forward_passes`：模型前向推理次数。当前为 `1`，表示每次视角只执行一次网络推理；增大它可能得到更多随机候选，但会增加计算时间。
- `max_retries`：当目标区域初次没有抓取候选时，最多重新采样多少次随机相机视角并重复推理。它不代表候选数量，也不改变单次模型输出的数量。
