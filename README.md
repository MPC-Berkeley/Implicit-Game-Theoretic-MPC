<div align="center">

# IGT-MPC: Implicit Game-Theoretic MPC
This repository contains the implementation of the paper <em>"Learning Multi-agent Motion Planning Strategies from Generalized Nash Equilibrium for Model Predictive Control"</em> submitted to 2025 Learning for Dynamics and Control Conference (L4DC) 

[Hansung Kim (hansung@berkeley.edu)](https://github.com/hansungkim98122) &emsp; [Edward L. Zhu (edward.zhu@plus.ai)](https://www.linkedin.com/in/edward-zhu/) &emsp; [Chang Seok Lim (cshigh22@berkeley.edu)](https://www.linkedin.com/in/kevin-lim-315b3b258/) &emsp; [Francesco Borrelli](https://me.berkeley.edu/people/francesco-borrelli/)   

![](https://img.shields.io/badge/language-python-blue)
<a href='https://arxiv.org/abs/2402.01116'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

# Demonstration Video:
[![Link to video](http://img.youtube.com/vi/Id_qpvGETjQ/hqdefault.jpg)](https://youtu.be/Id_qpvGETjQ)
</div>

# Clone repository
```
git clone https://github.com/MPC-Berkeley/Implicit-Game-Theoretic-MPC.git
```
# Two vehicle Head-to-Head Racing
## Install dependent libraries
```
cd head2head_racing/lib/mpclab_common
pip install -e .
```
```
cd head2head_racing/lib/mpclab_controllers
pip install -e .
```
```
cd head2head_racing/lib/mpclab_simulation
pip install -e .
```
The solver for MPC also requires hpipm_python package which you can download from here <href>https://github.com/giaf/hpipm/tree/master</href>

```
python evaluate.py
```
# Two vehicle Un-signalized Intersection Navigation
## Download the Game-Theoretic Interaction Dataset and unzip in 'intersection_navigation/game_theoretic_NN/dataset/':
https://drive.google.com/drive/folders/1_8X7iMNEwCyPxwwrzvA_sD0aoYWLmUq4?usp=drive_link

```
python evaluate.py --save_dir <save_directory> --eval_mode <mode: str> --sc <int>
```

1) Replace <save_directory> with a local directory
2) eval_mode: [gt_mpc,mpc]
3) sc: [1,2,3,4,5,6,7,8]
