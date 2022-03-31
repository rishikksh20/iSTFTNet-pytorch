# iSTFTNet : Fast and Lightweight Mel-spectrogram Vocoder Incorporating Inverse Short-time Fourier Transform
This repo try to implement [iSTFTNet : Fast and Lightweight Mel-spectrogram Vocoder Incorporating Inverse Short-time Fourier Transform](https://arxiv.org/pdf/2203.02395.pdf) specifically model `C8C8I`.
![](iSTFTnet.PNG)
`Disclaimer : This repo is build for testing purpose. The code is not optimized for performance.`
## Training :
```
python train.py --config config_v1.json
```

## Note:
* We are able to get good quality of audio with 30 % less training compared to original hifigan.
* This model approx 60 % faster than counterpart hifigan.

## Citations :
```
@inproceedings{kaneko2022istftnet,
title={{iSTFTNet}: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time Fourier Transform},
author={Takuhiro Kaneko and Kou Tanaka and Hirokazu Kameoka and Shogo Seki},
booktitle={ICASSP},
year={2022},
}
```

## References:
* https://github.com/jik876/hifi-gan
