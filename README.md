# choquet-layer-torch
 An implementation of the [Choquet Layer](https://github.com/Jacobiano/ChoquetLayer) for PyTorch.

Original code and paper by S. VELASCO-FORERO, CMM Mines Paris


### Notes
In PyTorch, you don't usually need to specify the input size of the entire model upfront due to its use of dynamic computation graphs. In the provided model, we use an AdaptiveAvgPool2d layer, which outputs a fixed-size vector regardless of the input size. This approach negates the need to specify the img_size parameter directly in the model architecture.