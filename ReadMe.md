FdSig
===

Very simple frequency domain(FFT based) image steganography/waterprint/signature.

Reference: [https://www.zhihu.com/question/50735753/answer/122593277]()

Support index shuffle with secret in image mode and text mode:

![](./sc_img.png)

![](./sc_text_secret.png)

Usage
---

usage: FdSig.py [-h] [-o OUTPUT] [-s SECRET] [-i IMAGESIGN] [-t TEXTSIGN]
                [-a ALPHA] [-d DECODE] [-v]
                file

positional arguments:
  file                  Original image filename.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output filename.
  -s SECRET, --secret SECRET
                        Secret to generate index mapping.
  -i IMAGESIGN, --image IMAGESIGN
                        Signature image filename.
  -t TEXTSIGN, --text TEXTSIGN
                        Signature text.
  -a ALPHA, --alpha ALPHA
                        Signature blending weight.
  -d DECODE, --decode DECODE
                        Image filename to be decoded.
  -v                    Display image.
