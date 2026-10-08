# Nut Assembly APIs

Nut Assembly currently has three supported control APIs. They share the same public robot actions (`goto_pose`, gripper control, and home motion), but differ in how they find the object and construct the target pose.

## `FrankaControlNutAssemblyGeometricApi`

Implementation: `capx/integrations/franka/nut_assembly_geometric.py`.

This is the Molmo-free pipeline. SAM3 performs text segmentation, with Nut-specific prompt fallbacks and geometric mask handling for the square nut, hollow center, and extruded handle. The selected mask is deprojected into 3-D; an oriented bounding box supplies the object orientation. The OBB Z axis is flipped when it points upward, but the remaining orientation is still geometry-driven and can be lateral for a handle. The API uses `init_pyroki_local(self._env)`, so it solves against the current MuJoCo model and does not require an HTTP IK service. It does not provide the Molmo point-query helper or the newer verify recorder inherited by Visual.

## `FrankaControlNutAssemblyVisualApi`

Implementation: `capx/integrations/franka/nut_assembly_visual.py`.

This is the main visual pipeline. Molmo produces a semantic 2-D point, and SAM3 uses that point as its prompt. The point and segmentation mask are used for depth deprojection; the mask point-cloud OBB remains the source of the returned orientation. It uses local MuJoCo IK, records SAM/Molmo images and IK position/quaternion diagnostics through the shared verify recorder, and now exposes `point_prompt_molmo(text_prompt)` directly. This is the replacement for the former standalone Molmo wrapper.

## `FrankaControlNutAssemblyGuideApi`

Implementation: `capx/integrations/franka/nut_assembly_guide.py`.

Guide inherits Visual completely for perception infrastructure, motion, local IK, gripper actions, and verification. It changes only pose construction: the Molmo pixel supplies the XY/deprojected point, while orientation is fixed to a world top-down pose whose tool-local +Z points toward world -Z. The world pose is converted to robot-base coordinates before being passed to local IK. This API is appropriate when grasp approach must be vertical and OBB orientation is not trusted.

## Configuration

All three names are registered in `capx/integrations/__init__.py` and can be selected independently in a task config:

```yaml
apis:
  - FrankaControlNutAssemblyGeometricApi
```

Use `FrankaControlNutAssemblyVisualApi` for Molmo plus OBB orientation, `FrankaControlNutAssemblyGuideApi` for Molmo plus fixed top-down orientation, and `FrankaControlNutAssemblyGeometricApi` when Molmo should not be used.
