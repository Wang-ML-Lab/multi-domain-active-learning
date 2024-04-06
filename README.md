# Composite Active Learning: <br>Towards Multi-Domain Active Learning with Theoretical Guarantees
This repo contains the code for our AAAI 2024 paper:<br>
**Composite Active Learning: Towards Multi-Domain Active Learning with Theoretical Guarantees**<br>
Guang-Yuan Hao, Hengguan Huang, Haotian Wang, Jie Gao, Hao Wang<br>
*AAAI 2024*<br>
[[Paper](https://arxiv.org/abs/2402.02110)] 

## Outline for This README
* [Brief Introduction for CAL](#brief-introduction-for-CAL)
* [Related Works](#also-check-our-relevant-work)
* [Reference](#reference)

## Brief Introduction for CAL
Active learning (AL) aims to improve model performance within a fixed labeling budget by choosing the most informative data points to label. 
Existing AL focuses on the single-domain setting, where all data come from the same domain (e.g., the same dataset). 
However, many real-world tasks often involve multiple domains. For example, in visual recognition, it is often desirable to train an image classifier that works across different environments (e.g., different backgrounds), where images from each environment constitute one domain. 
Such a multi-domain AL setting is challenging for prior methods because they (1) ignore the similarity among different domains when assigning labeling budget and (2) fail to handle distribution shift of data across different domains. 
In this paper, 
we propose the first general method, dubbed composite active learning (CAL), for multi-domain AL. 
Our approach explicitly considers the domain-level and instance-level information in the problem; 
CAL first assigns domain-level budgets according to domain-level importance, which is estimated by optimizing an upper error bound that we develop; 
with the domain-level budgets, CAL then leverages a certain instance-level query strategy to select samples to label from each domain. 
Our theoretical analysis shows that our method achieves a better error bound compared to current AL methods. Our empirical results demonstrate that our approach significantly outperforms the state-of-the-art AL methods on both synthetic and real-world multi-domain datasets.

## Installation
```python
conda env create -f environment.yml
```


## Also Check Our Relevant Work

<span id="paper_1">[1] **Taxonomy-Structured Domain Adaptation**<br>
Tianyi Liu*, Zihao Xu*, Hao He, Guang-Yuan Hao, Guang-He Lee, Hao Wang<br>
*Fortieth International Conference on Machine Learning (ICML), 2023*<br>
[[Paper](https://arxiv.org/abs/2306.07874)] [[OpenReview](https://openreview.net/forum?id=ybl9lzdZw7)] [[PPT](https://shsjxzh.github.io/files/TSDA_5_minutes.pdf)] [[Talk (Youtube)](https://www.youtube.com/watch?app=desktop&v=hRWfAsi0Uks)] [[Talk (Bilibili)](https://www.bilibili.com/video/BV13g4y1A7Uq/?spm_id_from=333.999.list.card_archive.click&vd_source=38c48d8008e903abbc6aa45a5cc63d8f)]
<br>*"\*" indicates equal contribution.*<br>

<span id="paper_2">[2] **Domain-Indexing Variational Bayes: Interpretable Domain Index for Domain Adaptation**<br>
Zihao Xu*, Guang-Yuan Hao*, Hao He, Hao Wang<br>
*Eleventh International Conference on Learning Representations, 2023*<br>
[[Paper](https://arxiv.org/abs/2302.02561)] [[OpenReview](https://openreview.net/forum?id=pxStyaf2oJ5&referrer=%5Bthe%20profile%20of%20Zihao%20Xu%5D(%2Fprofile%3Fid%3D~Zihao_Xu2))] [[PPT](https://shsjxzh.github.io/files/VDI_10_miniutes_2nd_version_to_pdf.pdf)] [[Talk (Youtube)](https://www.youtube.com/watch?v=xARD4VG19ec)] [[Talk (Bilibili)](https://www.bilibili.com/video/BV13N411w734/?vd_source=38c48d8008e903abbc6aa45a5cc63d8f)]


<span id="paper_3">[3] **Graph-Relational Domain Adaptation**<br></span>
Zihao Xu, Hao He, Guang-He Lee, Yuyang Wang, Hao Wang<br>
*Tenth International Conference on Learning Representations (ICLR), 2022*<br>
[[Paper](http://wanghao.in/paper/ICLR22_GRDA.pdf)] [[Code](https://github.com/Wang-ML-Lab/GRDA)] [[Talk](https://www.youtube.com/watch?v=oNM5hZGVv34)] [[Slides](http://wanghao.in/slides/GRDA_slides.pptx)]

<span id="paper_4">[4] **Continuously Indexed Domain Adaptation**<br></span>
Hao Wang*, Hao He*, Dina Katabi<br>
*Thirty-Seventh International Conference on Machine Learning (ICML), 2020*<br>
[[Paper](http://wanghao.in/paper/ICML20_CIDA.pdf)] [[Code](https://github.com/hehaodele/CIDA)] [[Talk](https://www.youtube.com/watch?v=KtZPSCD-WhQ)] [[Blog](http://wanghao.in/CIDA-Blog/CIDA.html)] [[Slides](http://wanghao.in/slides/CIDA_slides.pptx)] [[Website](http://cida.csail.mit.edu/)]

## Reference
[Composite Active Learning: Towards Multi-Domain Active Learning with Theoretical Guarantees](https://arxiv.org/abs/2402.02110)
```bib
@article{hao2024composite,
  title={Composite Active Learning: Towards Multi-Domain Active Learning with Theoretical Guarantees},
  author={Hao, Guang-Yuan and Huang, Hengguan and Wang, Haotian and Gao, Jie and Wang, Hao},
  journal={AAAI},
  year={2024}
}
```
