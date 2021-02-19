
import stillleben as sl

import pathlib

MY_PATH = pathlib.Path(__file__).parent.absolute()

_C = sl.extension.load(name='gripper_sim_C', sources=[MY_PATH / 'gripper_sim.cpp'], extra_cflags=['-g'], verbose=True)

GripperSim = _C.GripperSim
