# README

Source code of "Robust Transductive Few-shot Learning via Joint Message Passing and Prototype-based Soft-label Propagation".

## Quick Start

1. Download [pretrained.zip](https://drive.google.com/file/d/1jDKqG2A3mgv01LZmVC_BFIq_2JvkVSC6/view?usp=sharing)  and unzip it to get the pretrained folder, put it in the root directory, and put it on the same level as other folders
3. Modify the script folder name of `exp_scripts` in `main()` in main.py and select the appropriate script list
4. Right-click pycharm to run `main.py` or execute `python3 main.py`
5. The experimental results are saved in `/output/{exp_name}/{log_name}.log`

The first experiment needs to rebuild the task according to the data set, and the built task is saved in `/pretrained/preload`

## Citation
```
@article{wang2023robust,
  title={Robust Transductive Few-shot Learning via Joint Message Passing and Prototype-based Soft-label Propagation},
  author={Wang, Jiahui and Xu, Qin and Jiang, Bo and Luo, Bin},
  journal={arXiv preprint arXiv:2311.17096},
  year={2023}
}
```
