# View Discerning Networks for 3D shape recognition

Codebase for *Learning Discriminative 3D Shape Representations by View Discerning Networks* [[arXiv]](https://arxiv.org/abs/1808.03823). The goal of the project is to gauge the quality of projection views of 3D shapes, with weighted shape features for shape recognition task.

Here we offer the training and test code on ModelNet40 dataset for two structures for view discerning networks in the paper, `Channel-wise Score Unit(CSU)` and `Part-wise Score Unit(PSU)`.

## Codebase at a Glance
`dataset/`: Rendered views of shapes for 3D dataset such as ModelNet and ShapeNetCore55

`model/`: Prototxts of `Channel-wise Score Unit` and `Part-wise Score Unit`

`train_vdn/`: Training and test code for view discerning networks.


## Platform
Our code has been tested on Windows 10 with Matlab 2014a.

## Run

#### Setups
	1. Compile [CaffeMex_v2](https://github.com/sciencefans/CaffeMex_v2/) with matlab interface
	2. Add `CaffeMex_v2/matlab/` to matlab search path
	3. Configure the parameters in `train_vdn/train_vdn.m`, `train_vdn/test_vdn.m`, including the path of prototxt and the relative param. The pretrain_model can be found in here(https://github.com/lim0606/caffe-googlenet-bn).
	4. Place your rendered views of ModelNet40 dataset to the directory `dataset\` and running the `train_vdn\generate_ModelNet40_data.m` to generate your dataset split.
	
#### Training
	train_vdn/train_vdn.m
#### Test
	train_vdn/test_vdn.m



## Citation
Please kindly cite our work if it helps your research:

    @article{leng2018learning,
	  title={Learning Discriminative 3D Shape Representations by View Discerning Networks},
	  author={Leng, Biao and Zhang, Cheng and Zhou, Xiaochen and Xu, Cheng and Xu, Kai},
	  journal={IEEE transactions on visualization and computer graphics},
	  year={2018},
	  publisher={IEEE}
	}