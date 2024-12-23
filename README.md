# Frequency-Guided Iterative Network for Image Steganography (FIS)

The implementation of "Frequency-Guided Iterative Network for Image Steganography".



## Getting Started

### Prerequisites
#### Environment
- Python >= 3.6
- PyTorch >= 1.10.2
#### Datasets
Download the datasets [subsampled image datasets](https://drive.google.com/file/d/1ai9D3Z0lcdEnRX24pUL_XfuFSjWtbh5K) (same as LISO). It is the subset of [Div2k](https://data.vision.ee.ethz.ch/cvl/DIV2K/), [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), and [MS COCO](https://cocodataset.org). To use them, set your dataset path in the 'utils.py'.

### Training
```bash
python main.py --bits 2 --dataset div2k
```

### Evaluation
```bash
python main.py --eval --bits 2 --dataset div2k --load checkpoints/div2k/2_bits.steg
```


## Running Example
This is the running example of my project.
![example.png](example.png)


## Acknowledgements
- [LISO](https://github.com/cxy1997/LISO)
- [CBAM](https://github.com/Jongchan/attention-module)
- [TrajGRU](https://github.com/FunnyEngineer/TrajGRU)
- [Extensive Reading During Research](https://xinch3nwang.github.io/extensive_reading_fis.html)
