# This file contains code adapted from:
#
# 1. SMPLSim (https://github.com/ZhengyiLuo/SMPLSim)
#    Copyright (c) 2024 Zhengyi Luo
#    Licensed under the BSD 3-Clause License.

import numpy as np
import mujoco

def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_connector (mujoco 3.x API)
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    # mujoco 3.x uses different API signature for mjv_connector
    mujoco.mjv_connector(scene.geoms[scene.ngeom-1],
                         mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                         np.array(point1, dtype=np.float64),
                         np.array(point2, dtype=np.float64))