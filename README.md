# Hyperparameter Selection using Gradient-Free Optimization Algorithms

## 1. Introduction
Hyperparameters play a crucial role in the performance of machine learning models. Selecting the optimal set of hyperparameters is a challenging task that often requires an exhaustive search of the hyperparameter space. In this report, we investigate the performance of two gradient-free optimization algorithms, namely Pattern Search and Nesterov Random Search (RS), for ML hyperparameter selection. We compare these algorithms with a grid search, which serves as our baseline for comparison.

## 2. Methodology
We utilized the CIFAR10 dataset to train a Convolutional Neural Network (CNN) model for image classification. The hyperparameters under consideration were the SGD mini-batch size and the Adam α, β1, and β2 values. We performed a grid search to systematically explore the hyperparameter space. The ML loss function was plotted against optimization iterations, and the set of hyperparameters yielding the lowest loss value was chosen as the best performing.

Pattern Search and Nesterov Random Search require appropriate parameter selection. We investigated the impact of the initial set of hyperparameters, the range of hyperparameters, and the number of iterations on the performance of these algorithms. For Pattern Search, we experimented with different initial sets of hyperparameters: random, the hyperparameters selected by the grid search, and the hyperparameters selected by Pattern Search.

## 3. Evaluation
After training and evaluating the CNN model using the hyperparameters selected by Pattern Search and Random Search algorithms, we compared their performance. We specifically compared the validation accuracy, which represents the percentage of correctly classified images in the validation set.

Additionally, we examined the impact of the parameters of Pattern Search and Random Search on their performance. For Pattern Search, we investigated the impact of the initial set of hyperparameters, the range of hyperparameters, and the number of iterations on the validation accuracy. Similarly, for Random Search, we investigated the impact of the range of hyperparameters and the number of iterations on the validation accuracy.

## 4. Results and Analysis
We applied Nesterov Random Search and Pattern Search algorithms to both SGD and Adam optimizers, tuning their respective hyperparameters. We trained the models with the best selected hyperparameters and evaluated their performance.

### 4.1 Nesterov Random Search
#### 4.1.1 SGD Optimizer
We used a grid of learning rates [1e-2, 1e-3, 1e-4] and momentums [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]. Using Nesterov Random Search, we obtained the best hyperparameters in 255 seconds. The learning curve plot showed that the validation accuracy increased up to 8 epochs and remained relatively constant thereafter. The loss decreased up to 6 epochs and then remained constant. The overall accuracy on the train and test datasets was 72% and 49%, respectively.

#### 4.1.2 Adam Optimizer
We used a grid of learning rates [1e-2, 1e-3, 1e-4], beta_1 values [0.8, 0.9, 0.95, 0.99], and beta_2 values [0.8, 0.9, 0.95, 0.99]. Nesterov Random Search found the best hyperparameters in 250 seconds. The learning curve plot showed that the validation accuracy increased up to 13 epochs, started decreasing, and then remained constant. The loss decreased up to 6 epochs and then remained constant. The overall accuracy on the train and test datasets was 67% and 49%, respectively.

### 4.2 Pattern Search
#### 4.2.1 SGD Optimizer
For SGD optimizer, we used parameters including learning rates [1e-2, 1e-3, 1e-4] and momentums [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]. We applied Pattern Search algorithm with a step size of 0.01 for tuning and it took 496 seconds. The learning curve plot showed that the validation accuracy increased up to 20 epochs, and the minimum loss was achieved at the 20th epoch. The overall accuracy on the train and test datasets was 44% and 39%, respectively.

#### 4.2.2 Adam Optimizer
For Adam optimizer, we used a grid of learning rates [1e-2, 1e-3, 1e-4], beta_1 values [0.8, 0.9, 0.95, 0.99], and beta_2 values [0.8, 0.9, 0.95, 0.99]. Using Pattern Search algorithm, we found the best hyperparameters in 496 seconds. The learning curve plot showed that the validation accuracy increased up to 20 epochs, started decreasing, and then remained constant. The loss decreased up to 6 epochs and then remained constant. The overall accuracy on the train and test datasets was 72% and 50%, respectively.

## 5. Results and Analysis
After evaluating both algorithms, we summarized the results in the following table:

| Algorithm           | Epochs | Batch Size | Train Accuracy (%) | Test Accuracy (%) | Computational Time (s) |
|---------------------|--------|------------|--------------------|-------------------|------------------------|
| Nesterov Random Search (SGD) | 20     | 128        | 72                 | 49                | 255                    |
| Nesterov Random Search (Adam) | 20     | 128        | 67                 | 49                | 250                    |
| Pattern Search (SGD)         | 20     | 128        | 44                 | 39                | 587                    |
| Pattern Search (Adam)        | 20     | 128        | 72                 | 50                | 573                    |

Based on the results, we can make the following observations:

- Search Strategy: Random search selects hyperparameters uniformly at random, while pattern search utilizes a deterministic search strategy based on patterns and intervals.
- Exploration vs. Exploitation: Random search favors exploration over exploitation, while pattern search aims to balance both aspects.
- Efficiency: Random search is more efficient in large and high-dimensional search spaces, while pattern search can be more efficient in small and low-dimensional search spaces.
- Convergence: Pattern search may take longer to converge as it requires more iterations to sufficiently explore the search space.
- Robustness: Random search is less sensitive to the choice of hyperparameters compared to pattern search.
- Reproducibility: Random search is more reproducible since it does not depend on a specific search pattern, whereas pattern search relies on the initial guess and chosen pattern.


