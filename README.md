# FDRL
Unofficial implementation of [Feature Decomposition and Reconstruction Learning for Effective Facial Expression Recognition - CVPR'21](https://openaccess.thecvf.com/content/CVPR2021/papers/Ruan_Feature_Decomposition_and_Reconstruction_Learning_for_Effective_Facial_Expression_Recognition_CVPR_2021_paper.pdf)

## Training
1. First download the pretrained resnet18 and put it in backbone folder
2. To train the model according to the paper use the following command:
```
python train.py --exp_name {name_for_exp}
```
I set the default parameters according to the paper, however I cannot reproduce the result. I use the following command to generate the best result (89.31) in terms of overall accuracy):
```
python train.py --lr 0.001 --lambda_2 0.001 --exp_name test
```
## Testing
Use the following command:
```
python eval.py --resume {path_to_trained_model_in_checkpoints}
```
Note that if you use other arguments at training, please make sure apply them at testing.

## Cite
```
@InProceedings{Ruan_2021_CVPR,
    author    = {Ruan, Delian and Yan, Yan and Lai, Shenqi and Chai, Zhenhua and Shen, Chunhua and Wang, Hanzi},
    title     = {Feature Decomposition and Reconstruction Learning for Effective Facial Expression Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {7660-7669}
}
```
If you find this repo useful, please star, tks.
