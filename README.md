# Textual and Visual Guided Task Adaptation for Source‑free Cross‑Domain Few‑Shot Segmentation
[[`Paper`](https://arxiv.org/abs/2508.05213v1)] accepted for ACMMM'25.
## Abstract

Few‑Shot Segmentation (FSS) facilitates the efficient segmentation of novel objects using only a limited number of labeled samples. However, its performance often deteriorates considerably when substantial domain discrepancies exist between training and deployment scenarios. To address this limitation, Cross‑Domain Few‑Shot Segmentation (CD‑FSS) has emerged as a significant research area, aiming to mitigate performance degradation caused by such domain shifts. While prior methods have demonstrated the feasibility of effective CD‑FSS without direct access to source data during testing, their exclusive reliance on a small number of support images for target domain adaptation frequently leads to overfitting and restricts their ability to capture intra‑class appearance variations.

In this work, we propose a source‑free CD‑FSS method that leverages both textual and visual information to facilitate target domain task adaptation without requiring source domain data. Specifically, we first append Task‑Specific Attention Adapters (TSAA) to the feature pyramid of a pretrained backbone, which adapt multi‑level features extracted from the shared pre‑trained backbone to the target task. Then, the parameters of the TSAA are trained through a Visual‑Visual Embedding Alignment (VVEA) module and a Text‑Visual Embedding Alignment (TVEA) module. The VVEA module utilizes global‑local visual features to align image features across different views, while the TVEA module leverages textual priors from pre‑aligned multi‑modal features (e.g., from CLIP) to guide cross‑modal adaptation. By combining the outputs of these modules through dense comparison operations and subsequent fusion via skip connections, our method produces refined binary query masks.

Under both 1‑shot and 5‑shot settings, the proposed approach achieves average segmentation accuracy improvements of 2.18 % and 4.11 %, respectively, across four cross‑domain datasets, significantly outperforming state‑of‑the‑art CD‑FSS methods. 

## Datasets

The following datasets are used for evaluation in CD‑FSS:

### Target domains:

* **Deepglobe:**  
  * Home: http://deepglobe.org/  
  * Direct: https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset  
  * Preprocessed Data: https://drive.google.com/file/d/10qsi1NRyFKFyoIq1gAKDab6xkbE0Vc74/view?usp=sharing

* **ISIC2018:**  
  * Home: http://challenge2018.isic-archive.com  
  * Direct (must login): https://challenge.isic-archive.com/data#2018

* **Chest X‑ray:**  
  * Home: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/  
  * Direct: https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels

* **FSS‑1000:**  
  * Home: https://github.com/HKUSTCV/FSS-1000  
  * Direct: https://drive.google.com/file/d/16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI/view
# TVGTANet

## 🥰 Acknowledgements
Our code is built upon the works of [ABCDFSS](https://github.com/Vision-Kek/ABCDFSS), [WeCLIP](https://github.com/zbf1991/WeCLIP), we appreciate the authors for their excellent contributions!


## 📝 Citation
If you use this code for your research or project, please consider citing our paper. Thanks!🥂:
```
@article{ACMMM2025TVGTANet,
  title={Textual and Visual Guided Task Adaptation for Source-Free Cross-Domain Few-Shot Segmentation},
  author={Jianming Liu, Wenlong Qiu, and Haitao Wei},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia (MM’25)},
  year={2025}
}
```
