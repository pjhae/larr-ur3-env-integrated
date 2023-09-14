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
![image](https://github.com/pjhae/larr-ur3-env-integrated/assets/74540268/984f1ca2-a779-4c13-bb67-784684c31aa8)



#### Parameter sensitivity for scale factor
![image](https://github.com/pjhae/larr-ur3-env-integrated/assets/74540268/7b424423-ceb4-44bb-b43c-c9bac984b7c0)

