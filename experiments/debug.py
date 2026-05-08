# 在 PyCharm 里新建一个 test_mp_debug.py 跑这个
import os, sys
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import sumo_rl
from sumo_rl.environment.observations import PedestrianObservationFunction

BASE_DIR = "C:/Users/ucemdc3/PycharmProjects/sumo-rl"

par_env = sumo_rl.parallel_env(
    net_file=os.path.join(BASE_DIR, "nets/2x2grid/01.net.xml"),
    route_file=os.path.join(BASE_DIR, "nets/2x2grid/02.rou.xml"),
    use_gui=False, num_seconds=100, delta_time=5, min_green=5,
    reward_fn="diff-waiting-time-with-pedestrian",
    observation_class=PedestrianObservationFunction,
    sumo_warnings=False,
)

obs, info = par_env.reset()
print("agents:", par_env.agents)
print("type(par_env):", type(par_env))

# Walk through wrappers
obj = par_env
for i in range(10):
    print(f"  layer {i}: {type(obj).__name__}", end="")
    if hasattr(obj, 'traffic_signals'):
        print(" ← HAS traffic_signals!")
        break
    if hasattr(obj, 'env'):
        obj = obj.env
        print()
    elif hasattr(obj, 'aec_env'):
        obj = obj.aec_env
        print()
    else:
        print(" ← END")
        break

par_env.close()