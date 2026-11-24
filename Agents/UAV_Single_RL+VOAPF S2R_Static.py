"""All-static S2R scene, sharing the controller's selectable time interval."""

from importlib import import_module


_S2R_UAV = import_module('Agents.UAV_Single_RL+VOAPF S2R').UAV


class UAV(_S2R_UAV):
    """Keep every physical obstacle stationary and retain virtual point sources."""

    def __init__(self, param: dict, env=None) -> None:
        if env is not None:
            # Enforce the static variant even when selected in a mixed-scene XML.
            env.dynamic_enabled = False
            for obstacle in getattr(env, 'buildings', []):
                obstacle.clear_motion()
            env.dynamic_buildings = []
        super().__init__(param, env)
