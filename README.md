Custom OpenAI Gym
=================

OpenAI Gym based simulated(MuJoCo) and real environments for Universal Robotics robots.

### Quick Start Guide

Download the [latest release](https://github.com/snu-larr/dual-ur3-env/releases/latest) and install via pip.

```
cd larr-ur3-env-integrated-<latest release version>
pip install '.[all]'
```

Read the original README from OpenAI for instructions on installing MuJoCo.


Single/Dual arm env
=================
For simulation, move to :

```
/gym_custom/envs/custom/[single or dual]_ur3_env.py
```

For real experiment, move to :

```
/gym_custom/envs/real/[single or dual]_ur3_env.py
```


SAC for both environments
=================
```
/sac_[single or dual]/main.py
```


(Optional) Cross Entropy Method(CEM)
=================
To minimize sim-to-real gap, we used CEM.
```
/cem_[single or dual]
```
#### Method
![image](https://github.com/pjhae/larr-ur3-env-integrated/assets/74540268/c68d689d-d268-440f-b3ea-8c046bd02ccc)

#### Parameter sensitivity for scale factor
![image](https://github.com/pjhae/larr-ur3-env-integrated/assets/74540268/dc0ba33a-a9a2-46ac-b2d7-f95a2143fb53)


