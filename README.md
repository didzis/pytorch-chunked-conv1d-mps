## Introduction

Conv1d on MPS devices with output channels > 65536 gives [wrong results](https://github.com/pytorch/pytorch/issues/129207) on macOS versions below 15.1.
Because of this bug PyTorch 2.5.0 and 2.5.1 disabled Conv1d on MPS devices with more than 65536 output channels. This limitation [may be lifted](https://github.com/pytorch/pytorch/pull/140726)
in next PyTorch release (2.5.2?) for macOS versions at least 15.1.
This is a serious performance limitation for many Text-to-Speech (TTS) systems running inference on MPS devices. This module is a workaround to this bug by splitting Conv1d and ConvTranspose1d
in smaller chunks and then stitching results together. A similar approach workaround [may also be added](https://github.com/sdatkinson/neural-amp-modeler/pull/506) to the next PyTorch release.

## Usage

Import in your project like below and it will automatically fallback to chunked versions or will use `torch.nn.Conv1d` and `torch.nn.ConvTranspose1d` if no workaround is required.

```python
from chunked_conv1d import Conv1d, ConvTranspose1d
```
