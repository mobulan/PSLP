# README

Source code of Transductive Few-shot Learning with Prototype-based Label Propagation by Iterative Graph Refinement.

## Standard operation method

1. Download pretrained.zip and unzip it to get the pretrained folder, put it in the root directory, and put it on the same level as other folders
3. Modify the script folder name of `exp_scripts` in `main()` in main.py and select the appropriate script list
4. Right-click pycharm to run `main.py` or execute `python3 main.py`
5. The experimental results are saved in `/output/{exp_name}/{log_name}.log`

The first experiment needs to rebuild the task according to the data set, and the built task is saved in `/pretrained/preload`