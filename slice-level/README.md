# Covid-19 Image Classification

Following papers are implemented using PyTorch.

- ResNet ([1512.03385](https://arxiv.org/abs/1512.03385))
- DenseNet ([1608.06993](https://arxiv.org/abs/1608.06993), [2001.02394](https://arxiv.org/abs/2001.02394))

## Requirements

- Python >= 3.7
- PyTorch >= 1.4.0
- torchvision
- pyqt

## Usage

```
python train.py --batch-size=32 --size=512 --arch=resnet50
```

```
python tset.py
```

## GUI

```
python gg.py
```