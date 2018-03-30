# Classification of MNIST Digits
The MNIST dataset (Modified National Institute of Standards and Technology dataset) is a database of handwritten digits (0 to 9) and this dataset is often known as the "Hello World" dataset of Computer Vision.

The dataset consists of:
- Training: 55,000
- Development: 5,000
- Test: 10,000
  
Each handwritten digit is grayscale image of dimension 28 x 28 pixels. Here are a few typical MNIST digits:

<img src="https://github.com/nagarajarchak/MNISTDigits/blob/master/Images/DigitImage5.png" height="33%" width="33%"> <img src="https://github.com/nagarajarchak/MNISTDigits/blob/master/Images/DigitImage8.png" height="33%" width="33%"> <img src="https://github.com/nagarajarchak/MNISTDigits/blob/master/Images/DigitImage9.png" height="33%" width="33%">

The dataset is fed to an ANN and Mini-Batch Neural Network and each of these use Gradient Descent, RMSProp and Adam optimizers with a ReLU activation function.

### Neural Network Computation Graph
<img src="https://github.com/nagarajarchak/MNISTDigits/blob/master/Images/Computation%20Graph.png" height="100%" width="100%">


> Hyperparameter Tuning

- Learning Rate: 0.001.
- Hidden Layers: 3
- Hidden Units: 100
- Number of Epochs: 50
- Batch Size: 500
- Number of Iterations: 1001

## Cross Entropy & Accuracies

### Artificial Neural Network
> Cross Entropy graphs for Gradient Descent, RMSProp and Adam Optimizers respectively.

<img src="https://github.com/nagarajarchak/MNISTDigits/blob/master/Images/ANN%20Gradient%20Descent.png" height="33%" width="33%"> <img src="https://github.com/nagarajarchak/MNISTDigits/blob/master/Images/ANN%20RMSProp.png" height="33%" width="33%"> <img src="https://github.com/nagarajarchak/MNISTDigits/blob/master/Images/ANN%20Adam.png" height="33%" width="33%">

> Accuracies for ANN

| ANN | Gradient Descent | RMSProp | Adam |
| :---: | :---: | :---: | :---: |
| Train | 0.567927 | 0.998673 | 0.999982 |
| Development | 0.583400 | 0.977000 | 0.972800 |
| Test | 0.577800 | 0.977800 | 0.973000 |

### Mini-Batch Neural Network
> Cross Entropy graphs for Gradient Descent, RMSProp and Adam Optimizers respectively.

<img src="https://github.com/nagarajarchak/MNISTDigits/blob/master/Images/Cost%20graph%20-%20%20Gradient%20Descent%20Batch.png" height="33%" width="33%"> <img src="https://github.com/nagarajarchak/MNISTDigits/blob/master/Images/Cost%20Graph%20-%20RMSProp%20Batch.png" height="33%" width="33%"> <img src="https://github.com/nagarajarchak/MNISTDigits/blob/master/Images/Cost%20Graph%20-%20Adam%20Batches.png" height="33%" width="33%">

> Accuracies for Mini-Batch NN

| Mini-Batch | Gradient Descent | RMSProp | Adam |
| :---: | :---: | :---: | :---: |
| Train | 0.856364 | 0.999964 | 1.000000 |
| Development | 0.860400 | 0.978600 | 0.978800 |
| Test | 0.864500 | 0.978000 | 0.979000 |

*All graphs and images are generated using Tensorboard.*
