# Session 9 Assignment

**Goal is to create a network that**
1. has the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) 
2. total RF must be more than 44
3. one of the layers must use Depthwise Separable Convolution
4. one of the layers must use Dilated Convolution
5. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
6. use argumentation library and apply:
    - horizontal flip
    - shiftScaleRotate
    - coarseDropout (max_holes = 1, max_height=16px, max_width=16px, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
7. achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.

------
## model.py

The file contains the network as desired in the assignment (C1C2C3C40).

- Layer 3 of Block C1 is a Dilated convolution with dilation = 2.
- Layer 1 of Block C2 & C3 both are Depthwise Separable Convolutions.

The network has ~140k trainable parameters.

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             448
       BatchNorm2d-2           [-1, 16, 32, 32]              32
         Dropout2d-3           [-1, 16, 32, 32]               0
            Conv2d-4           [-1, 16, 32, 32]           2,320
       BatchNorm2d-5           [-1, 16, 32, 32]              32
         Dropout2d-6           [-1, 16, 32, 32]               0
            Conv2d-7           [-1, 16, 32, 32]           2,320
       BatchNorm2d-8           [-1, 16, 32, 32]              32
         Dropout2d-9           [-1, 16, 32, 32]               0
           Conv2d-10           [-1, 16, 32, 32]             160
           Conv2d-11           [-1, 32, 32, 32]             544
      BatchNorm2d-12           [-1, 32, 32, 32]              64
        Dropout2d-13           [-1, 32, 32, 32]               0
           Conv2d-14           [-1, 32, 32, 32]           9,248
      BatchNorm2d-15           [-1, 32, 32, 32]              64
        Dropout2d-16           [-1, 32, 32, 32]               0
           Conv2d-17           [-1, 32, 16, 16]           9,248
      BatchNorm2d-18           [-1, 32, 16, 16]              64
        Dropout2d-19           [-1, 32, 16, 16]               0
           Conv2d-20           [-1, 32, 16, 16]             320
           Conv2d-21           [-1, 64, 16, 16]           2,112
      BatchNorm2d-22           [-1, 64, 16, 16]             128
        Dropout2d-23           [-1, 64, 16, 16]               0
           Conv2d-24           [-1, 64, 16, 16]          36,928
      BatchNorm2d-25           [-1, 64, 16, 16]             128
        Dropout2d-26           [-1, 64, 16, 16]               0
           Conv2d-27             [-1, 64, 8, 8]          36,928
      BatchNorm2d-28             [-1, 64, 8, 8]             128
        Dropout2d-29             [-1, 64, 8, 8]               0
           Conv2d-30             [-1, 32, 8, 8]          18,464
      BatchNorm2d-31             [-1, 32, 8, 8]              64
        Dropout2d-32             [-1, 32, 8, 8]               0
           Conv2d-33             [-1, 32, 8, 8]           9,248
      BatchNorm2d-34             [-1, 32, 8, 8]              64
        Dropout2d-35             [-1, 32, 8, 8]               0
           Conv2d-36             [-1, 32, 6, 6]           9,248
        AvgPool2d-37             [-1, 32, 1, 1]               0
           Conv2d-38             [-1, 10, 1, 1]             330
================================================================
Total params: 138,666
Trainable params: 138,666
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.95
Params size (MB): 0.53
Estimated Total Size (MB): 4.49
----------------------------------------------------------------
```

## transforms.py
The file contains albumentations trasforms which are applied to the input dataset CIFAR10.
```
train_transforms = A.Compose(
        [
            A.Normalize(mean, std),
            A.HorizontalFlip(p=p),
            A.ShiftScaleRotate(p=p),
            A.CoarseDropout(max_holes = 1,
                            max_height=16,
                            max_width=16,
                            min_holes = 1,
                            min_height=16,
                            min_width=16,
                            fill_value=(mean),
                            mask_fill_value = None,
                            p=p
            )
        ]
    )
```

## dataset.py
CustomCIFAR10Dataset is created on top of CIFAR10 to take care of albumentation transform.
```
class CustomCIFAR10Dataset(Dataset):
    def __init__(self, root_dir='../data', train=True, transform=None):
        self.transform = transform
        self.dataset = datasets.CIFAR10(root_dir, train=train, download=True)
        self.root_dir = root_dir


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        data = self.dataset[index][0]
        target = self.dataset[index][1]

        img = np.array(data)

        if self.transform:
            augmentations = self.transform(image=img)
            img = augmentations["image"]

        target = torch.from_numpy(np.array(target))
        img = torch.from_numpy(img.transpose(2, 0, 1))

        return img, target
```

## utils.py
The file contains utility & helper functions needed for training & for evaluating our model.

## S9.ipynb
The file is an IPython notebook. The notebook imports model, functions from utils.py, dataset.py, model.py etc.

```
Epoch 1
Train: Loss=1.3965 Batch_id=781 Accuracy=36.58: 100%|██████████| 782/782 [06:12<00:00,  2.10it/s]
Test set: Average loss: 1.3362, Accuracy: 5074/10000 (50.74%)

Epoch 2
Train: Loss=1.7470 Batch_id=781 Accuracy=49.38: 100%|██████████| 782/782 [05:38<00:00,  2.31it/s]
Test set: Average loss: 1.1368, Accuracy: 5916/10000 (59.16%)

Epoch 3
Train: Loss=1.0874 Batch_id=781 Accuracy=55.47: 100%|██████████| 782/782 [05:36<00:00,  2.32it/s]
Test set: Average loss: 1.0408, Accuracy: 6283/10000 (62.83%)

Epoch 4
Train: Loss=2.1104 Batch_id=781 Accuracy=60.19: 100%|██████████| 782/782 [05:54<00:00,  2.20it/s]
Test set: Average loss: 0.9429, Accuracy: 6749/10000 (67.49%)

Epoch 5
Train: Loss=0.9271 Batch_id=781 Accuracy=63.43: 100%|██████████| 782/782 [05:55<00:00,  2.20it/s]
Test set: Average loss: 0.8184, Accuracy: 7197/10000 (71.97%)

Epoch 6
Train: Loss=1.0048 Batch_id=781 Accuracy=65.30: 100%|██████████| 782/782 [05:28<00:00,  2.38it/s]
Test set: Average loss: 0.7589, Accuracy: 7360/10000 (73.60%)

Epoch 7
Train: Loss=0.9277 Batch_id=781 Accuracy=66.80: 100%|██████████| 782/782 [05:47<00:00,  2.25it/s]
Test set: Average loss: 0.7493, Accuracy: 7382/10000 (73.82%)

Epoch 8
Train: Loss=1.3592 Batch_id=781 Accuracy=68.47: 100%|██████████| 782/782 [05:49<00:00,  2.24it/s]
Test set: Average loss: 0.6919, Accuracy: 7676/10000 (76.76%)

Epoch 9
Train: Loss=1.5236 Batch_id=781 Accuracy=69.52: 100%|██████████| 782/782 [05:45<00:00,  2.26it/s]
Test set: Average loss: 0.6420, Accuracy: 7810/10000 (78.10%)

Epoch 10
Train: Loss=0.8150 Batch_id=781 Accuracy=70.72: 100%|██████████| 782/782 [06:15<00:00,  2.08it/s]
Test set: Average loss: 0.6334, Accuracy: 7834/10000 (78.34%)

Epoch 11
Train: Loss=1.5170 Batch_id=781 Accuracy=70.99: 100%|██████████| 782/782 [06:22<00:00,  2.04it/s]
Test set: Average loss: 0.6044, Accuracy: 7952/10000 (79.52%)

Epoch 12
Train: Loss=1.0824 Batch_id=781 Accuracy=71.73: 100%|██████████| 782/782 [06:14<00:00,  2.09it/s]
Test set: Average loss: 0.6052, Accuracy: 7937/10000 (79.37%)

Epoch 13
Train: Loss=1.0829 Batch_id=781 Accuracy=73.14: 100%|██████████| 782/782 [06:16<00:00,  2.08it/s]
Test set: Average loss: 0.5669, Accuracy: 8098/10000 (80.98%)

Epoch 14
Train: Loss=0.6474 Batch_id=781 Accuracy=73.19: 100%|██████████| 782/782 [06:20<00:00,  2.06it/s]
Test set: Average loss: 0.5568, Accuracy: 8133/10000 (81.33%)

Epoch 15
Train: Loss=0.6355 Batch_id=781 Accuracy=73.71: 100%|██████████| 782/782 [06:27<00:00,  2.02it/s]
Test set: Average loss: 0.5264, Accuracy: 8216/10000 (82.16%)

Epoch 16
Train: Loss=0.6642 Batch_id=781 Accuracy=74.36: 100%|██████████| 782/782 [06:35<00:00,  1.98it/s]
Test set: Average loss: 0.5292, Accuracy: 8202/10000 (82.02%)

Epoch 17
Train: Loss=0.6756 Batch_id=781 Accuracy=74.93: 100%|██████████| 782/782 [06:43<00:00,  1.94it/s]
Test set: Average loss: 0.5514, Accuracy: 8159/10000 (81.59%)

Epoch 00017: reducing learning rate of group 0 to 1.0000e-03.
Epoch 18
Train: Loss=0.4190 Batch_id=781 Accuracy=76.91: 100%|██████████| 782/782 [06:30<00:00,  2.00it/s]
Test set: Average loss: 0.4809, Accuracy: 8351/10000 (83.51%)

Epoch 19
Train: Loss=0.6412 Batch_id=781 Accuracy=77.62: 100%|██████████| 782/782 [06:29<00:00,  2.01it/s]
Test set: Average loss: 0.4665, Accuracy: 8378/10000 (83.78%)

Epoch 20
Train: Loss=0.6554 Batch_id=781 Accuracy=77.45: 100%|██████████| 782/782 [06:32<00:00,  1.99it/s]
Test set: Average loss: 0.4624, Accuracy: 8419/10000 (84.19%)

Epoch 21
Train: Loss=0.5678 Batch_id=781 Accuracy=77.71: 100%|██████████| 782/782 [06:41<00:00,  1.95it/s]
Test set: Average loss: 0.4584, Accuracy: 8441/10000 (84.41%)

Epoch 22
Train: Loss=0.3720 Batch_id=781 Accuracy=77.89: 100%|██████████| 782/782 [06:44<00:00,  1.93it/s]
Test set: Average loss: 0.4562, Accuracy: 8428/10000 (84.28%)

Epoch 23
Train: Loss=0.7143 Batch_id=781 Accuracy=77.98: 100%|██████████| 782/782 [07:04<00:00,  1.84it/s]
Test set: Average loss: 0.4597, Accuracy: 8420/10000 (84.20%)

Epoch 24
Train: Loss=1.0993 Batch_id=781 Accuracy=78.40: 100%|██████████| 782/782 [07:01<00:00,  1.85it/s]
Test set: Average loss: 0.4518, Accuracy: 8462/10000 (84.62%)

Epoch 25
Train: Loss=1.1225 Batch_id=781 Accuracy=78.50: 100%|██████████| 782/782 [07:02<00:00,  1.85it/s]
Test set: Average loss: 0.4514, Accuracy: 8442/10000 (84.42%)

Epoch 26
Train: Loss=1.1547 Batch_id=781 Accuracy=78.66: 100%|██████████| 782/782 [06:57<00:00,  1.87it/s]
Test set: Average loss: 0.4530, Accuracy: 8468/10000 (84.68%)

Epoch 27
Train: Loss=0.8737 Batch_id=781 Accuracy=78.20: 100%|██████████| 782/782 [07:06<00:00,  1.84it/s]
Test set: Average loss: 0.4501, Accuracy: 8440/10000 (84.40%)

Epoch 28
Train: Loss=0.3743 Batch_id=781 Accuracy=78.56: 100%|██████████| 782/782 [07:04<00:00,  1.84it/s]
Test set: Average loss: 0.4472, Accuracy: 8473/10000 (84.73%)

Epoch 29
Train: Loss=0.5673 Batch_id=781 Accuracy=78.41: 100%|██████████| 782/782 [07:13<00:00,  1.80it/s]
Test set: Average loss: 0.4499, Accuracy: 8460/10000 (84.60%)

Epoch 30
Train: Loss=0.5143 Batch_id=781 Accuracy=79.28: 100%|██████████| 782/782 [39:56<00:00,  3.06s/it]    
Test set: Average loss: 0.4439, Accuracy: 8490/10000 (84.90%)

Epoch 31
Train: Loss=0.7731 Batch_id=781 Accuracy=79.15: 100%|██████████| 782/782 [06:22<00:00,  2.04it/s]
Test set: Average loss: 0.4401, Accuracy: 8518/10000 (85.18%)
```

## How to setup
### Prerequisits
```
1. python 3.8 or higher
2. pip 22 or higher
```

It's recommended to use virtualenv so that there's no conflict of package versions if there are multiple projects configured on a single system. 
Read more about [virtualenv](https://virtualenv.pypa.io/en/latest/). 

Once virtualenv is activated (or otherwise not opted), install required packages using following command. 

```
pip install requirements.txt
```

## Running IPython Notebook using jupyter
To run the notebook locally -
```
$> cd <to the project folder>
$> jupyter notebook
```
The jupyter server starts with the following output -
```
To access the notebook, open this file in a browser:
        file:///<path to home folder>/Library/Jupyter/runtime/nbserver-71178-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/?token=64bfa8105e212068866f24d651f51d2b1d4cc6e2627fad41
     or http://127.0.0.1:8888/?token=64bfa8105e212068866f24d651f51d2b1d4cc6e2627fad41
```

Open the above link in your favourite browser, a page similar to below shall be loaded.

![Jupyter server index page](https://github.com/piygr/s5erav1/assets/135162847/40087757-4c99-4b98-8abd-5c4ce95eda38)

- Click on the notebook (.ipynb) link.

A page similar to below shall be loaded. Make sure, it shows *trusted* in top bar. 
If it's not _trusted_, click on *Trust* button and add to the trusted files.

![Jupyter notebook page](https://github.com/piygr/s5erav1/assets/135162847/7858da8f-e07e-47cd-9aa9-19c8c569def1)
Now, the notebook can be operated from the action panel.

Happy Modeling :-) 
 
