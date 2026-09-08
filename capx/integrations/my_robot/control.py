# capx/integrations/my_robot/control.py
from capx.integrations.base_api import ApiBase

class MyRobotControlApi(ApiBase):
    def __init__(self, env):
        super().__init__(env)
        # Initialize any perception/planning services

    def functions(self):
        """Return the dict of functions exposed to LLM-generated code.

        Each function's signature and docstring will be shown to the model.
        Write clear, complete docstrings — they ARE the model's documentation.
        """
        return {
            "move_to": self.move_to,
            "grasp": self.grasp,
            "get_object_position": self.get_object_position,
        }

    def move_to(self, position: np.ndarray, orientation: np.ndarray) -> None:
        """Move the robot end-effector to a target pose.

        Args:
            position: (3,) XYZ target in meters, world frame.
            orientation: (4,) WXYZ unit quaternion.

        Returns:
            None
        """
        # Implementation using self._env (the low-level simulator)
        ...

    def get_object_position(self, object_name: str) -> np.ndarray:
        """Get the 3D position of a named object.

        Args:
            object_name: Natural language name of the object (e.g., "red cube").

        Returns:
            position: (3,) XYZ in meters, world frame.
        """
        ...