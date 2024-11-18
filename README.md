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
## Download the Game-Theoretic Interaction Dataset and unzip in :
https://drive.google.com/drive/folders/1_8X7iMNEwCyPxwwrzvA_sD0aoYWLmUq4?usp=drive_link

```
python evaluate.py --save_dir <save_directory> --eval_mode <mode: str> --sc <int>
```

1) Replace <save_directory> with a local directory
2) eval_mode: [gt_mpc,mpc]
3) sc: [1,2,3,4,5,6,7,8]
