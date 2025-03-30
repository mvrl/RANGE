# RANGE: Retrieval Augmented Neural Fields for Multi-Resolution Geo-Embeddings (CVPR 2025) 🌎🌏🌍
<div align="center">

[![Static Badge](https://img.shields.io/badge/2502.19781-red?label=arxiv)](https://arxiv.org/abs/2502.19781)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow
)](https://huggingface.co/collections/MVRL/range-67e99fa1dfc6c86a3b872c09)

</center>

[Aayush Dhakal*](https://sites.wustl.edu/aayush/)&nbsp;&nbsp;&nbsp;
[Srikumar Sastry](https://vishu26.github.io/)&nbsp;&nbsp;&nbsp;
[Subash Khanal](https://subash-khanal.github.io/)&nbsp;&nbsp;&nbsp;
[Eric Xing](https://ericx003.github.io/)&nbsp;&nbsp;&nbsp;
[Adeel Ahmad](https://adealgis.wixsite.com/adeel-ahmad-geog)&nbsp;&nbsp;&nbsp;
[Nathan Jacobs](https://jacobsn.github.io/)


</div>
<br>
<br>
This repository is the official implementation of [RANGE](https://arxiv.org/abs/2502.19781).
RANGE (Retrieval Augmented Neural Fields for Multi-Resolution Geo-Embeddings) is a retrieval-augmented framework for embedding geographic coordinates. RANGE directly estimates the visual features for a given location, allowing the representations to capture high-resolution information. 

![](images/framework_cam.jpg)

## 🏋️‍♂️ Performance on Downstream Tasks
We showed through a large number of downstream tasks that RANGE embeddings outperform several state-of-the-art location embedding methods such as SatCLIP, GeoCLIP, CSP, etc.
![](images/downstream.png)

## 🔥 Multi-scale Geoembeddings
Our method enforces a spatial smoothness constraint. Manipulating this constraint allows generating geo-embeddings at desired frequencies.
![](images/beta_interpolation_2.png)

## ⚙️ Usage
The required model weights and embeddings are made available in huggingface.



📑 Citation

```bibtex
@article{dhakal2025range,
  title={RANGE: Retrieval Augmented Neural Fields for Multi-Resolution Geo-Embeddings},
  author={Dhakal, Aayush and Sastry, Srikumar and Khanal, Subash and Ahmad, Adeel and Xing, Eric and Jacobs, Nathan},
  booktitle={Computer Vision and Pattern Recognition},
  year={2025},
  organization={IEEE/CVF}
}
```


## 🔍 Additional Links
Check out our lab website for other interesting works on geospatial understanding and mapping:
* [Multi-Modal Vision Research Lab (MVRL)](https://mvrl.cse.wustl.edu/)
* [Related Works from MVRL](https://mvrl.cse.wustl.edu/publications/)
