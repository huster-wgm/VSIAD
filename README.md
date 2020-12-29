# "Learn to Recover Visible Color for Video Surveillance in a Day" (ECCV2020 Oral)

This repository contains the code and dataset for the paper ["Learn to Recover Visible Color for Video Surveillance in a Day"]() by Guangming Wu, Yinqiang Zheng, Zhiling Guo, et al.


## Requirements

The source code have been valified on the following env:

- NVIDIA GPU > 8GB
- Ubuntu 16.04 LTS
- Anaconda Python(3.6)
- [Pytorch](https://pytorch.org/) > 1.0
- OpenCV 3

## VSIAD Dataset

```
Link => https://univtokyo-my.sharepoint.com/:u:/g/personal/9291958590_utac_u-tokyo_ac_jp/EZTYb_6LBWhEv1je1gzT8-wB1Dud678ewydOS_wM1YawdA?e=8Y37Kx
passwd => eccv2020
```

## Pretrained model
```
checkppoint/
--res18ynet-VC24-RGB2LAB_iter_100000.pth                 => (SSN, SSM => 0)
--res18ynetsync-VC24-L2LAB_iter_100000.pth               => (SSN, L2LAB)
--res18ynetsync-VC24-RGB2LAB_iter_100000.pth             => (SSN, full model)
--res18ynetsync-VC24-RGB2LAB-nop_iter_100000.pth.pth     => (SSN, perceptual => 0)
--res18ynetsync-VC24-LAB2LAB_iter_100000.pth             => (SSN, LAB2LAB)
--res18ynetsyncGN-VC24-RGB2LAB_iter_100000.pth           => (SSN, IN2GN)
```

## Usage

```
pending
```

## Publication
G. Wu, Y. Zheng, Z. Guo, et al., &quot;Learn to Recover Visible Color for Video Surveillance in a Day,&quot; <i>European Conference on Computer Vision (**ECCV**)</i>, to appear, 2020. (<font color="blue">oral presentation, acceptance rate: 2%</font>)  


## Citation

```
@inproceedings{wu2020learn,
  title={Learn to Recover Visible Color for Video Surveillance in a Day},
  author={Wu, Guangming and Zheng, Yinqiang and Guo, Zhiling and Cai, Zekun and Shi, Xiaodan and Ding, Xin and Huang, Yifei and Guo, Yimin and Shibasaki, Ryosuke},
  booktitle={European Conference on Computer Vision},
  pages={495--511},
  year={2020},
  organization={Springer}
}
```

## Contact
For any question, please contact
```
huster-wgm(.at.)csis.u-tokyo.ac.jp or
guangmingwu2010(.at.)gmail.com
```

