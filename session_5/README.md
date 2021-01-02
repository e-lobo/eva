## Session 5 Assignment

# [Step 1](step_1/README.md) :

## Target:

1.   Set up a working model with Gap , 1x1 convolution , dropout & batch norm

## Results:

1.   Parameters: 1,970,272
2.   Best Training Accuracy: 100%
3.   Best Test Accuracy: 99.57%

## Conclusion 

*   Model is over-fitting
*   Less than 10k params used.
*   Max testing accuracy achieved and can never increase since 100% accuracy achieved in training.
*   Model has too many params.

# [Step 2](step_2/README.md) :

## Target:

## Results:

## Conclusion

# [Step 3](step_3/README.md) :

## Target:

1.   Addition of Image Augmentation (Random Rotation).
2.   Addition of GAP (Global Average Pooling) layer to the network.

## Results:


1.   Parameters: 10,308
2.   Best Training Accuracy: 99.26%
3.   Best Test Accuracy: 99.31%

## Conclusion 

*   Model is over-fitting

# [Step 4](step_4/README.md) :

## Target:

1. Addition of Image Augmentation (Random Rotation).
2. Addition of GAP (Global Average Pooling) layer to the network.
3. Addition of scheduler.

## Results:

1. Parameters: 8,016
2. Best Training Accuracy: 99.27
3. Best Test Accuracy: 99.41%

## Conclusion

* Model is not over-fitting
* Less than 10k params used.
* Accuracy rising gradually and we observe 99.4% accuracy