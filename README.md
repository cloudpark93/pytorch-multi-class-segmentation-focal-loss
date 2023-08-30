# Focal loss for the Multi-class semantic segmentation in PyTorch
PyTorch implementation of focal loss for multi-class semantic segmentation

I personally prefer to use the non-alpha form focal loss for multi-class semantic segmentation.

If you prefer to use the alpha form focal loss, please prepare a 

```python
...
fn_loss = FocalLoss()
