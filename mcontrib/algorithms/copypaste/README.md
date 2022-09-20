# ✂️ CopyPaste

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

`Computer Vision`

CopyPaste is a data augmentation technique for image segmentation tasks that randomly pastes object instances onto an image. A pair of source and target samples are randomly chosen from a batch of data and a set of randomly chosen (without replacement) instances are extracted from the source sample. The selected instances are then transformed and pasted into the target sample.
This augmentation method can serve as a regularization technique to enhance the generalization capability of segmentation models for computer vision. This was tested on composer version 0.9.0.



| ![CopyPaste](https://storage.googleapis.com/docs.mosaicml.com/images/methods/copypaste.png) |
|:--:
|*An example of data augmentation using CopyPaste. Object instances are randomly chosen from a source sample, jittered, and pasted into a target instance.*|

## How to Use

### Functional Interface

Here we run `CopyPaste` directly on a batch
```python
# Run the CopyPaste algorithm directly on the batch data using the Composer functional API
import torch
import torch.nn.functional as F
from mcontrib.algorithms.copypaste import copypaste_batch

# Example configuration for the copypaste_batch, checkout the CopyPaste docstring for more details
copypaste_configs = {
    "p": 1.0,
    "max_copied_instances": None,
    "area_threshold": 100,
    "padding_factor": 0.5,
    "jitter_scale": (0.01, 0.99),
    "jitter_ratio": (1.0, 1.0),
    "p_flip": 1.0,
    "bg_color": 0
}

def training_loop(model, train_loader, num_epochs):
    opt = torch.optim.Adam(model.parameters())

    model.train()

    for epoch in range(num_epochs):
        for X, y in train_loader:
            X_copypaste, y_copypaste = copypaste_batch(X, y, configs=copypaste_configs)
            y_hat = model(X_copypaste)
            loss = F.cross_entropy(y_hat, y_copypaste)
            loss.backward()
            opt.step()
            opt.zero_grad()
```

### Composer Trainer

```python
# Instantiate the algorithm and pass it into the Trainer
# The trainer will automatically run it at the appropriate points in the training loop

from composer.algorithms import CopyPaste
from composer.trainer import Trainer

copypaste = CopyPaste()

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration='1ep',
    algorithms=[copypaste]
)

trainer.fit()
```

### Implementation Details

Our implementation of CopyPaste augmentation is aligned with the [CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf) by Ghiasi et al. as we choose the strategy of randomly choosing instances from a source sample and pasting them into random locations of the target sample.

The process begins with randomly drawing a pair of source and target samples (with replacement) from the data batch. Each sample contains an image of shape `(C, H, W)` as well as its corresponding annotated mask of shape `(H, W)` with pixel values denoting the class ID. A set of randomly chosen object instances are selected from the source sample and random transformations (Large-scale Jittering) are applied to each of the instances. Jittered intances are then pasted into a random location of the target image.

## Suggested Hyperparameters

Our CopyPaste implementation offers a variety of tunable hyperparameters which enable the user to control the mechanics of transformations that are applied to the copied instances as well as governing the statistics of stochastic parts of this augmentation method. However, the most impactful hyperparameter seems to be the probability of applying CopyPaste to a selected pair of source & target samples. This tunable parameter is set to `p = 0.5` by default to ensure feeding both original (non-augmented) and augmented samples to the network during a training job.

## Technical Details

During the process of copying instances from the source sample to the target sample, several considerations are applied to the to-be-copied instance.
If the mask of a copied instance collides with an existing mask in the target instance, the copied mask is placed over the existing mask.

After randomly drawing an instance from the source sample, the instance goes through Large-scale Jittering, which includes horizontal flipping, rescaling, and cropping. Statistics of these stochastic transformations are defined by the configurable hyperparameters.

> ❗ CopyPaste ignores instances that are too small
>
> After copying an instance from the source sample, a set of randomly configured transformations are applied to that instance. If the resulting mask/object is smaller than a threshold, the jittered instance (both in RGB domain and in the annotation) will not be pasted in the target image.

CopyPaste is intended to improve generalization performance, and we empirically found this to be the case in our semantic segmentation settings. The original paper also reports improvements in mask accuracy (IoU) and classification performance.


> 🚧 Composing Regularization Methods
>
> As general rule, composing regularization methods may lead to diminishing returns in quality improvements. CopyPaste is one such regularization method.

Data augmentation techniques can sometimes put additional load on the CPU, potentially to the point where the CPU becomes a bottleneck for training.
To prevent this from happening, our implementation of CopyPaste mainly takes place on the GPU.
Doing so avoids putting additional work on the CPU (since augmentation occurs on the GPU).

> 🚧 CopyPaste Requires a Small Amount of Additional GPU Compute and Memory
>
> CopyPaste requires a small amount of additional GPU compute and memory to produce the augmented batch.
> In our experiments, we have found these additional resource requirements to be negligible.

## Attribution

[_Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation_](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf) by Golnaz Ghiasi, Yin Cui, Aravind Srinivas, Rui Qian, Tsung-Yi Lin, Ekin D. Cubuk, Quoc V. Le, Barret Zoph. Published in CVPR 2021.

*This Composer implementation of this method and the accompanying documentation were produced by the Behrad Toghi at MosaicML.*