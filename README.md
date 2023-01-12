# Important

This repo is a **fork** of [mil-ad/snip](https://github.com/mil-ad/snip) that I modified to my needs. **The actual implementation is not mine**, but Milad Alizadeh's. I just adapted it to my needs. The corresponding paper to the code is by [Lee et al. (2018)](https://arxiv.org/abs/1810.02340).

## Changes I made 

For my master's thesis , I rewrote the SNIP method in a more efficient way that makes use of the prune function that is built in to PyTorch. I'm using this code as a sanity-checking method, to compare my results  see whether my implementation works. To be able to do this, I applied the following changes to this repo: 

* Moved loss to `crossentropy`
* Moved to `timm` for models
* Moved to my dataloaders for fair comparison
* Included a stylized print function for all layers
* Removed anything to do with training as I don't need it
* Added a main to demonstrate the functionality