# Focal loss for the Multi-class semantic segmentation in PyTorch
PyTorch implementation of focal loss for multi-class semantic segmentation.  

If you want to use the alpha form focal loss, you need to do two things:
1. Please prepare **a set of alpha** for each class.  
2. Change the comment in the code as below:  
```python
focal_loss = self.alpha[targets] * (1 - pt)**self.gamma * ce_loss
#focal_loss = (1 - pt) ** self.gamma * ce_loss
```


# Non-alpha form Focal loss
```python
...
fn_loss = FocalLoss()

pred = model(x)
loss = fn_loss(pred, target)

...
```

# Alpha form Focal loss
```python
...
class_weights = [a set of alpha for each class]
fn_loss = FocalLoss(alpha = class_weights)

pred = model(x)
loss = fn_loss(pred, target)

...
```

# Extra
Please do visit my [**colleague's**](https://github.com/jinsoo9595) github as well!  
https://github.com/jinsoo9595
