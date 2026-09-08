Input is a raw MuJoCo observation captured in sam_test/round1_scene.
depth_m.npy is metric depth in meters and intrinsics.npy is the camera K.
The mask is SAM3's top text-segmentation result, not a SAM2 artifact.
strict uses the runtime filter_grasps=True setting; unfiltered is a
diagnostic only, used to identify candidates removed by that filter.
