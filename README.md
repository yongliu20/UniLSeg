# Universal Segmentation at Arbitrary Granularity with Language Instruction
Yong Liu, Cairong Zhang, Yitong Wang, Jiahao Wang, Yujiu Yang, Yansong Tang

The repository contains the official implementation of "Universal Segmentation at Arbitrary Granularity with Language Instruction"[CVPR 2024]

[Paper](https://arxiv.org/abs/2312.01623)

<a href='https://arxiv.org/abs/2312.01623'><img src='https://img.shields.io/badge/ArXiv-2312.01623-red'></a> 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/universal-segmentation-at-arbitrary/referring-expression-segmentation-on-refcoco-3)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco-3?p=universal-segmentation-at-arbitrary)
 <br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/universal-segmentation-at-arbitrary/referring-expression-segmentation-on-refcoco-4)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco-4?p=universal-segmentation-at-arbitrary)
 <br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/universal-segmentation-at-arbitrary/referring-expression-segmentation-on-refcoco-5)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco-5?p=universal-segmentation-at-arbitrary)
 <br>
 [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/universal-segmentation-at-arbitrary/referring-expression-segmentation-on-refcocog)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcocog?p=universal-segmentation-at-arbitrary)
  <br>
  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/universal-segmentation-at-arbitrary/referring-expression-segmentation-on-refcocog-1)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcocog-1?p=universal-segmentation-at-arbitrary)





## 📖 Abstract
This paper aims to achieve universal segmentation of arbitrary semantic level.
Despite significant progress in recent years, specialist segmentation approaches are limited to specific tasks and data distribution. Retraining a new model for adaptation to new scenarios or settings takes expensive computation and time cost, which raises the demand for versatile and universal segmentation model that can cater to various granularity. 
Although some attempts have been made for unifying different segmentation tasks or generalization to various scenarios, limitations in the definition of paradigms and input-output spaces make it difficult for them to achieve accurate understanding of content at arbitrary granularity. 
To this end, we present UniLSeg, a universal segmentation model that can perform segmentation at any semantic level with the guidance of language instructions. 
For training UniLSeg, we reorganize a group of tasks from original diverse distributions into a unified data format, where images with texts describing segmentation targets as input and corresponding masks are output. Combined with a automatic annotation engine for utilizing numerous unlabeled data, UniLSeg achieves excellent performance on various tasks and settings, surpassing both specialist and unified segmentation models.

---
## 📖 Pipeline
<p align="center">
 <img src="imgs/teaser.png" width="100%">
</p>
<p align="center">
 <img src="imgs/pipeline.png" width="100%">
</p>


We have open-sourced the general inference code and UniLSeg-20 model weights (w/o finetuned on specified task dataset). If you find any bugs due to carelessness on our part in organizing the code, feel free to contact us and point that!







### Installation
Install required packages. 

```bash
conda create -n UniLSeg python=3.7
conda activate UniLSeg
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
pip install -r requirements.txt
```



### Usage

- #### Pretrained Weight
  We have provided the pretrained UniLSeg-20 model weights (w/o finetuned on specified task dataset) and other pre-trained backbone weights. Please download them from [here](https://drive.google.com/drive/folders/1llKmPaOUhsAqxtopFdfnIBAlspvw_I4I?usp=drive_link) and put them under the current path. 



#### General Inference 
You can run the general inference by the following command:

  ```
 python general_inference.py  --img <IMG_PATH> --exp <'EXPRESSION'> --sp <MASK_SAVE_PATH>
  ```



### Cite 

If you find our work helpful, we'd appreciate it if you could cite our paper in your work.
```
@article{liu2023universal,
  title={Universal Segmentation at Arbitrary Granularity with Language Instruction},
  author={Liu, Yong and Zhang, Cairong and Wang, Yitong and Wang, Jiahao and Yang, Yujiu and Tang, Yansong},
  journal={arXiv preprint arXiv:2312.01623},
  year={2023}
}
```
