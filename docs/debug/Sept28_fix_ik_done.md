这些相似差异主要不是 Nut API 中人为写入的固定误差，而是两部分叠加：

  1. API 明确加入的目标偏移

  在 capx/integrations/franka/nut_assembly_visual.py 的 goto_pose() 中：

  offset_pos = pos + rot.apply(self._TCP_OFFSET)

  其中：

  _TCP_OFFSET = [0.0, 0.0, -0.107]

  所以 CSV 中记录的 target_* 不是原始 goto_pose(position)，而是补偿 TCP 后的末端目标。

  当传入 z_approach 时，还会进一步计算：

  z_offset_pos = offset_pos + rot.apply([0, 0, -z_approach])

  因此同一个 goto_pose 可能写两行：

  phase=approach
  phase=final

  这会造成 target 之间出现规律性的约 0.05 m 或 0.08 m 差异。这个差异是 API 设计出来的，是正常的 approach 轨迹。

  2. 实际位置重复，导致误差重复

  CSV 中 actual 位置只有少数几个聚类，例如：

  [0.05015, 0.00000, 1.47462]
  [0.05142, -0.00008, 1.46957]
  [0.07392, -0.00002, 1.09458]

  但 target 有很多不同值。因此相似误差来自：

  不同 target
      -> 实际执行后落在同一个 actual
      -> actual - target 形成一组相似模式

  这不是 API 固定加出的差异，而是执行链路没有到达不同目标后产生的结果。

  当前执行逻辑是：

  self._env.move_to_joints_blocking(joints)

  而 move_to_joints_blocking() 使用：

  tolerance = 0.02
  max_steps = 100

  它最多执行 100 个仿真步。如果关节误差仍未小于 0.02，就会直接退出，不会报错。此时 API 立即读取当前末端位置写入 CSV：

  self._record_ik_vector(
      offset_pos,
      self._sim_ee_pose(),
      joints,
      phase="final",
  )

  所以 CSV 记录的是：

  请求目标 target
  控制器停止时的实际位置 actual

  而不是 PyRoKI 内部 FK 计算出来的位置。

  目前还有两个数据质量问题：

  - 这份 CSV 有 97 行，但只有 44 行包含新增统计字段，前面的旧记录是在表头升级前生成的。
  - VDM 多次生成或重复执行代码，导致很多 goto_pose 调用本身就是重复的，不代表 97 个独立目标。

  因此应区分三种差异：

  target 之间约 0.05/0.08 m 的变化
  => goto_pose 的 approach/TCP offset 逻辑

  相同 actual 对应多个不同 target
  => 控制器未到位、最大步数耗尽或关节目标未执行完全

  actual 整体落在 z≈1.1 或 z≈1.47，而 target z≈0.02~0.45
  => 更可能是 PyRoKI 坐标系/URDF 基座与仿真世界坐标不一致

  最后一类才是当前最严重的问题。_TCP_OFFSET 和 z_approach 只能解释厘米级目标变化，不能解释约 1 米的系统性偏差。Mink warning 也不能解释这种模式。

  下一步最有价值的是比较新增字段：

  returned_joint_*
  actual_*

  并对 returned_joint_* 做仿真 FK。如果 FK 位置已经接近 target，但 actual 仍远离 target，问题在关节控制执行；如果 FK 本身就落在 z≈1.1/1.47，问题在 PyRoKI 的 URDF 坐标系或目标坐标转换。