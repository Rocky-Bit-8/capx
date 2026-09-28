# Nut Assembly 抓取姿态逻辑分析

当前任务配置注册的是 `FrankaControlNutAssemblyVisualApi`，因此 `sample_grasp_pose()` 不会执行普通 `FrankaControlApi` 中的 Contact-GraspNet 流程。Visual API 的 `sample_grasp_pose()` 只是直接调用 `get_object_pose(object_name)` 并原样返回位置和四元数。换言之，它的目标是估计被查询物体/局部区域的视觉 pose，并没有独立的“夹爪抓取姿态生成”阶段，也没有根据夹爪宽度、接近方向或碰撞情况对 grasp candidates 排序。

`get_object_pose()` 的流程是：先用 Molmo/SAM 得到目标像素和分割 mask，再结合深度和相机内参反投影为点云；随后对分割点云计算 Open3D oriented bounding box。返回位置不是 OBB 中心，而是 Molmo 点位按 mask 最小深度反投影并转换到 world 后的位置；返回旋转则来自整片分割点云的 OBB。点云在相机坐标系求 OBB，其旋转矩阵随后与相机外参组合，转换到 world frame，作为 `obb_tf_world.wxyz_xyz[:4]` 返回。也就是说，位置和旋转分别由 mask 内的最近深度点与整片点云的几何包围盒决定，来源并不完全相同。代码只做了一项方向修正：如果 OBB 的 Z 轴在 world 中指向上，就右乘一个绕 X 轴 180 度的旋转，使该轴朝下。除此之外，没有把姿态约束为“从上方抓取”，也没有要求夹爪的 world Z 轴严格接近 `[0, 0, -1]`。

因此，OBB 的主轴排序和分割点云形状会直接影响四元数。对于带有横向手柄或细长结构的 nut，OBB 可能将物体的横向几何轴作为夹爪的主要朝向；仅翻转 Z 轴并不能消除绕竖直轴的自由度，也不能修正夹爪坐标系与物体坐标系的轴对应关系。`sample_grasp_pose()` 也没有用 Molmo 的 point 去构造夹爪姿态：Molmo point 只决定目标位置，四元数仍由 mask 点云 OBB 决定。最终 `goto_pose()` 把收到的 WXYZ 四元数直接作为 IK 目标姿态；它根据该四元数旋转 TCP offset，并将姿态与位置一同交给 local IK。若指定 `z_approach`，approach 位移也沿目标工具坐标系的局部 Z 轴计算，而不是固定沿 world Z。因此，横向倾斜的 target 不仅会令末端横向接近，还会改变 approach 轨迹。

需要特别区分：`get_object_pose()` 中的 `fixed_rotation` 只对名称同时包含 `square` 和 `block` 的目标使用仿真提供的方块姿态；nut 或 handle 不满足该条件，通常使用 OBB 姿态。普通 Control API 对 cube 有单独的固定四元数分支，但当前 nut Visual API 没有该分支。后续修复应在返回 grasp pose 前显式构造 top-down 姿态，或将 OBB 姿态投影到 world 竖直约束下，同时保留合理的平面内 yaw；还应明确哪个末端局部轴对应夹爪接近轴/夹爪开合轴，并用 world-frame 三轴向量验证，而不是仅凭四元数分量判断。当前问题更像是视觉 pose 被直接复用为 grasp pose，不能据此归因于 IK 求解器。
