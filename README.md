# Spatial Information Refinement for Chroma Intra Prediction in Video Coding

## Abstract
Video compression benefits from advanced chroma intra prediction methods, such as the Cross-Component Linear Model (CCLM) which uses linear models to approximate the relationship between the luma and chroma components. Recently it has been proven that advanced cross-component prediction methods based on Neural Networks (NN) can bring additional coding gains. In this paper, spatial information refinement is proposed for improving NN-based chroma intra prediction. Specifically, the performance of chroma intra prediction can be improved by refined down-sampling or by incorporating location information. Experimental results show that the two proposed methods obtain 0.31%, 2.64%, 2.02% and 0.33%, 3.00%, 2.12% BD-rate reduction on Y, Cb and Cr components, respectively, under All-Intra configuration, when implemented in Versatile Video Coding (H.266/VVC) test model.

![visualisation-fig]

[visualisation-fig]: https://github.com/Chengyi-Zou/intra-chroma-attentionCNN-refinement/blob/main/visualisation/network.png

## Proposed method

Building upon the attention-based neural network in [this paper](https://ieeexplore.ieee.org/document/9292660), two schemes for spatial information refinement are proposed to improve the chroma prediction performance. By adding down-sampling branches or location information to the input, the network performance is improved as more accurate prediction values can be obtained. Without changing the main structure, the two proposed schemes enhance the cross-component boundary branch and the luma convolutional branch. The proposed two schemes are:

**Scheme A**: Adding down-sample branch.

**Scheme B**: Adding location map 

## Publication

Find the paper discribing our work on [arXiv](https://arxiv.org/abs/2109.11913).

Please cite with the following Bibtex code:

```
@ARTICLE{2021arXiv210911913Z,
       author = {{Zou}, Chengyi and {Wan}, Shuai and {Ji}, Tiannan and {Mrak}, Marta and {Gorriz Blanch}, Marc and {Herranz}, Luis},
        title = "{Spatial Information Refinement for Chroma Intra Prediction in Video Coding}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Multimedia},
         year = 2021,
        month = sep,
          eid = {arXiv:2109.11913},
        pages = {arXiv:2109.11913},
archivePrefix = {arXiv},
       eprint = {2109.11913},
 primaryClass = {cs.MM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv210911913Z},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## How to use

### Dependencies

The model is implemented in [Keras](https://github.com/fchollet/keras/tree/master/keras), which at its time is developed over [TensorFlow](https://www.tensorflow.org). Also, this code should be compatible with Python 3.6. And the reference code can be found in [this repository](https://github.com/bbc/intra-chroma-attentionCNN).

### Prepare data

Training examples were extracted from the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/), which contains high-definition high-resolution content of large diversity. This database contains 800 training samples and 100 samples for validation, providing 6 lower resolution versions with down-sampling by  factors of 2, 3 and 4 with a bilinear and unknown filters. For each data instance, one resolution was randomly selected and then M blocks of each NxN sizes (N=4, 8, 16) were chosen, making balanced sets between block sizes and uniformed spatial selections within each image. For the method of adding down-sample branch, we convert the picture from PNG format to YUV4:2:0 format, and for the method of adding location map, we convert the picture from PNG format to YUV4:4:4 format.  

Training and validation images are organised in 7 resolution classes. We expect the directory structure to be the following:

```
path/to/DIV2K/
  train/
    0/ # HR: 0001.png - 0800.png
    1/ # LR_bicubic_X2: 0001.png - 0800.png
    2/ # LR_unknown_X2: 0001.png - 0800.png
    3/ # LR_bicubic_X3: 0001.png - 0800.png
    4/ # LR_unknown_X3: 0001.png - 0800.png
    5/ # LR_bicubic_X4: 0001.png - 0800.png
    6/ # LR_unknown_X4: 0001.png - 0800.png
  val/
    0/ # HR: 0801.png - 0900.png
    1/ # LR_bicubic_X2: 0801.png - 0900.png
    2/ # LR_unknown_X2: 0801.png - 0900.png
    3/ # LR_bicubic_X3: 0801.png - 0900.png
    4/ # LR_unknown_X3: 0801.png - 0900.png
    5/ # LR_bicubic_X4: 0801.png - 0900.png
    6/ # LR_unknown_X4: 0801.png - 0900.png
```

To create random training and validation blocks of the desired resolution run:

```
python create_database[scheme].py -i path/to/DIV2K -o path/to/blocks
```

### Train a model configuration

To train a model run the ```train.py``` script selecting the desired configuration. Update the size-dependent configurations at ```config/att/``` and the multi-models at ```config/att_multi/```:

```
python train.py -c [path/to/cf_file].py -g [gpu number]
```

### Deploy a model scheme

In order to integrate the trained models into VTM 7.0, we need to export their parameters and apply the proposed simplifications. As explained in the paper, 3 multi-model schemes are considered, to deploy its parameters update the deployment config file at ```config/att_multi/``` and run:

```
python deploy[scheme].py -c config/deploy/scheme[X].py
```

The resultant weights and bias will be stored in the deploy path defined in the config file. In order to integrate them into the codec follow the next section to compile the updated VTM-7.0 version and copy the deployed arrays in ```VVCSoftware_VTM/source/Lib/CommonLib/NNIntraChromaPrediction.h```.

### Update VTM-7.0 with the proposed schemes

In order to generate a VTM-7.0 updated version with the proposed schemes, clone the original version and apply the patch differences relative to each scheme located at ```VTM-7.0-schemes/scheme[X].patch```:

```
git clone -b VTM-7.0 https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git
cd VVCSoftware_VTM
git apply ../VTM-7.0-schemes/scheme[X].patch
```

To compile the generated VTM-7.0 version follow the official instructions in ```VVCSoftware_VTM/README.md```.

### Reproduce the results

All the schemes are evaluated against a constrained VTM-7.0 anchor, whereby the VVC partitioning process is limited to using only square blocks of 4, 8 and 16 pixels. In order to generate the constrained VTM-7.0 anchor in this paper, apply the patch difference located at ```VTM-7.0-schemes/square_anchor.patch```.

## Contact

If you have any general doubt about our work or code which may be of interest for other researchers, please drop us an e-mail at <cyzou@mail.nwpu.edu.cn>.

