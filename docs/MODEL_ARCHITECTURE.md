# PhishNet Model Architecture Documentation

## Overview

PhishNet uses an ensemble of 15 diverse machine learning models per feature type (URL, DNS, WHOIS), totaling 45 models. This document provides comprehensive details on each model, their characteristics, and suitability for phishing detection.

---

## Model Selection Criteria

Models were selected based on:
1. **Learning Paradigm Diversity**: Tree-based, linear, probabilistic, distance-based, neural
2. **Industry Adoption**: Proven success in security/fraud detection domains
3. **Research Backing**: Published papers demonstrating effectiveness
4. **Computational Efficiency**: Balance between accuracy and inference speed
5. **Feature Type Suitability**: Effectiveness on URL structure, DNS records, and WHOIS metadata

---

## 1. Tree-Based Models (7 models)

### 1.1 Random Forest (RF)
**Algorithm**: Ensemble of decision trees using bagging with feature randomness

**How It Works**:
- Builds multiple decision trees on bootstrapped samples
- Each split considers random subset of features
- Final prediction via majority voting

**Strengths**:
- Robust to outliers and noise
- Handles non-linear relationships
- Built-in feature importance
- Minimal hyperparameter tuning needed

**Weaknesses**:
- Can be slower than boosting for inference
- May overfit on very noisy data
- Memory intensive (stores multiple trees)

**Best Use Cases**:
- **URL Features**: Excellent for capturing complex patterns in URL structure (length, special chars, entropy)
- **DNS Features**: Good for non-linear relationships in record counts and TTL values
- **WHOIS Features**: Effective for categorical features (privacy protection, registrar patterns)

**Industry Usage**:
- Microsoft (malware detection)
- PayPal (fraud detection)
- Cisco Umbrella (DNS threat detection)

**Research Backing**:
- Breiman (2001): "Random Forests" - original paper
- Ma et al. (2009): "Beyond Blacklists: Learning to Detect Malicious Web Sites from Suspicious URLs"

**Configuration**:
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
```

---

### 1.2 Extra Trees (Extremely Randomized Trees)
**Algorithm**: Like Random Forest but with fully random splits

**How It Works**:
- Similar to RF but splits are fully random (not optimized)
- More randomness = better variance reduction
- Faster training than Random Forest

**Strengths**:
- Even more robust to overfitting than RF
- Faster training (no split optimization)
- Better variance reduction

**Weaknesses**:
- May underfit if trees too random
- Slightly lower bias reduction than RF

**Best Use Cases**:
- **URL Features**: Excellent when dataset has high variance
- **DNS Features**: Good for noisy DNS records with inconsistent patterns
- **WHOIS Features**: Effective when WHOIS data quality varies

**Industry Usage**:
- Google (ad click fraud)
- Banking sector (credit card fraud)

**Research Backing**:
- Geurts et al. (2006): "Extremely randomized trees"
- Frequently used in Kaggle competitions for tabular data

**Configuration**:
```python
ExtraTreesClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
```

---

### 1.3 Gradient Boosting (GB)
**Algorithm**: Sequential ensemble that builds trees to correct previous errors

**How It Works**:
- Builds trees sequentially
- Each tree fits residual errors from previous ensemble
- Uses gradient descent to minimize loss

**Strengths**:
- High accuracy (often best single model)
- Handles mixed feature types well
- Good feature importance

**Weaknesses**:
- Slower training (sequential)
- Can overfit without careful tuning
- Sensitive to noisy data

**Best Use Cases**:
- **URL Features**: Excellent for structured URL patterns
- **DNS Features**: Strong on numeric DNS features (TTL, record counts)
- **WHOIS Features**: Good for age/registration patterns

**Industry Usage**:
- Yandex (search ranking)
- Zillow (home price prediction)
- Many Kaggle winners

**Research Backing**:
- Friedman (2001): "Greedy Function Approximation: A Gradient Boosting Machine"
- Chen et al. (2015): XGBoost paper cites original GB extensively

**Configuration**:
```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
```

---

### 1.4 Histogram Gradient Boosting (HistGB)
**Algorithm**: Fast gradient boosting using histogram-based splits

**How It Works**:
- Bins continuous features into discrete bins
- Uses histograms for faster split finding
- Native support for missing values

**Strengths**:
- Much faster than standard GB
- Native NaN handling (no imputation needed)
- Memory efficient
- Handles large datasets well

**Weaknesses**:
- May lose some precision from binning
- Newer algorithm (less battle-tested)

**Best Use Cases**:
- **URL Features**: Fast training on large URL datasets
- **DNS Features**: Excellent - handles NaN in DNS failures naturally
- **WHOIS Features**: Perfect - WHOIS often has missing data

**Industry Usage**:
- LightGBM and XGBoost both use histogram-based methods
- Widely adopted in production ML systems

**Research Backing**:
- Inspired by LightGBM's histogram approach
- Sklearn implementation (Pedregosa et al., 2011)

**Configuration**:
```python
HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
```

---

### 1.5 XGBoost (Extreme Gradient Boosting)
**Algorithm**: Highly optimized gradient boosting with regularization

**How It Works**:
- Advanced gradient boosting with L1/L2 regularization
- Uses second-order gradients (Newton-Raphson)
- Tree pruning and parallel processing

**Strengths**:
- State-of-the-art accuracy
- Built-in regularization prevents overfitting
- Fast (parallelized)
- Handles missing values

**Weaknesses**:
- Many hyperparameters to tune
- Can be overkill for simple problems
- Memory intensive

**Best Use Cases**:
- **URL Features**: Excellent for all URL features
- **DNS Features**: Top performer on structured DNS data
- **WHOIS Features**: Strong on categorical + numeric mix

**Industry Usage**:
- Airbnb (pricing)
- Uber (ETA prediction)
- Netflix (recommendation)
- Microsoft (Bing ranking)

**Research Backing**:
- Chen & Guestrin (2016): "XGBoost: A Scalable Tree Boosting System"
- Most popular algorithm in Kaggle competitions (2015-2018)

**Configuration**:
```python
XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=1,
    random_state=42,
    eval_metric='logloss'
)
```

---

### 1.6 LightGBM (Light Gradient Boosting Machine)
**Algorithm**: Ultra-fast gradient boosting using leaf-wise growth

**How It Works**:
- Grows trees leaf-wise (not level-wise)
- Uses histogram binning
- Gradient-based One-Side Sampling (GOSS)

**Strengths**:
- Fastest gradient boosting
- Lower memory usage
- Handles large datasets efficiently
- Good with categorical features

**Weaknesses**:
- Can overfit on small datasets
- Leaf-wise growth may create deep trees

**Best Use Cases**:
- **URL Features**: Fast training on large URL datasets
- **DNS Features**: Excellent for large-scale DNS analysis
- **WHOIS Features**: Good categorical feature handling

**Industry Usage**:
- Microsoft (Azure ML)
- Yandex (search ranking)
- Many production systems needing fast training

**Research Backing**:
- Ke et al. (2017): "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
- Used in winning solutions for many ML competitions

**Configuration**:
```python
LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)
```

---

### 1.7 CatBoost (Categorical Boosting)
**Algorithm**: Gradient boosting optimized for categorical features

**How It Works**:
- Ordered Target Statistics for categorical encoding
- Symmetric trees
- Built-in handling of categorical features

**Strengths**:
- Best-in-class categorical feature handling
- Robust to overfitting
- No need for extensive preprocessing
- Good default parameters

**Weaknesses**:
- Slower than LightGBM
- Less documentation than XGBoost

**Best Use Cases**:
- **URL Features**: Good for categorical URL components (TLD, scheme)
- **DNS Features**: Strong on record types (categorical)
- **WHOIS Features**: Excellent - many categorical features (registrar, privacy)

**Industry Usage**:
- Yandex (developed by Yandex)
- CERN (physics data analysis)
- Financial institutions (credit scoring)

**Research Backing**:
- Prokhorenkova et al. (2018): "CatBoost: unbiased boosting with categorical features"
- Competitive with XGBoost/LightGBM in benchmarks

**Configuration**:
```python
CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=5,
    random_state=42,
    verbose=0
)
```

---

## 2. Linear Models (4 models)

### 2.1 Logistic Regression L2 (Ridge)
**Algorithm**: Linear model with L2 regularization

**How It Works**:
- Learns linear decision boundary: P(y=1) = sigmoid(w^T x + b)
- L2 penalty shrinks coefficients toward zero
- Closed-form or gradient-based solution

**Strengths**:
- Fast training and inference
- Interpretable (coefficient = feature importance)
- Good baseline model
- Probabilistic outputs

**Weaknesses**:
- Assumes linear separability
- May underfit complex patterns
- Requires feature scaling

**Best Use Cases**:
- **URL Features**: Good for linear patterns (e.g., length thresholds)
- **DNS Features**: Effective for simple DNS anomalies
- **WHOIS Features**: Strong on age-based features (linear relationship)

**Industry Usage**:
- Widely used as baseline
- Production systems needing interpretability
- Real-time systems (fast inference)

**Research Backing**:
- Cox (1958): Original logistic regression
- Ng & Jordan (2002): "On Discriminative vs. Generative Classifiers"

**Configuration**:
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        penalty='l2',
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ))
])
```

---

### 2.2 Logistic Regression ElasticNet (L1 + L2)
**Algorithm**: Linear model with L1 and L2 regularization

**How It Works**:
- Combines L1 (Lasso) and L2 (Ridge) penalties
- L1 does feature selection (sets some weights to 0)
- L2 prevents overfitting on correlated features

**Strengths**:
- Automatic feature selection
- Handles correlated features better than pure L1
- More robust than L2-only
- Sparse solutions

**Weaknesses**:
- Slower than L2-only
- Two hyperparameters (alpha, l1_ratio)

**Best Use Cases**:
- **URL Features**: Excellent - many correlated URL features
- **DNS Features**: Good when many DNS features redundant
- **WHOIS Features**: Strong for feature selection from 49 features

**Industry Usage**:
- Text classification (sparse features)
- Genomics (high-dimensional data)
- Feature engineering pipelines

**Research Backing**:
- Zou & Hastie (2005): "Regularization and variable selection via the elastic net"

**Configuration**:
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0.5,
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ))
])
```

---

### 2.3 SGD Logistic (Stochastic Gradient Descent)
**Algorithm**: Online linear classifier using SGD optimization

**How It Works**:
- Updates weights incrementally on mini-batches
- Efficient for large datasets (doesn't load all data)
- Can use various loss functions

**Strengths**:
- Extremely fast on large datasets
- Memory efficient (online learning)
- Can be updated incrementally
- Good for streaming data

**Weaknesses**:
- Sensitive to feature scaling
- Requires tuning learning rate
- Less stable than batch methods

**Best Use Cases**:
- **URL Features**: Good for large-scale URL analysis
- **DNS Features**: Effective for continuous DNS monitoring
- **WHOIS Features**: Suitable for incremental WHOIS updates

**Industry Usage**:
- Large-scale text classification
- Online advertising (CTR prediction)
- Systems needing incremental updates

**Research Backing**:
- Bottou (2010): "Large-Scale Machine Learning with Stochastic Gradient Descent"
- Zhang (2004): "Solving large scale linear prediction problems using SGD"

**Configuration**:
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=0.0001,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ))
])
```

---

### 2.4 Linear SVM Calibrated
**Algorithm**: Linear Support Vector Machine with probability calibration

**How It Works**:
- Finds maximum-margin hyperplane
- Hinge loss + L2 regularization
- Platt scaling for probability calibration

**Strengths**:
- Effective in high-dimensional spaces
- Good generalization
- Works well with sparse features
- Fast inference

**Weaknesses**:
- Not naturally probabilistic (needs calibration)
- Sensitive to feature scaling
- Slower training than logistic regression

**Best Use Cases**:
- **URL Features**: Strong on high-dimensional URL features
- **DNS Features**: Good for structured DNS patterns
- **WHOIS Features**: Effective on categorical features (one-hot encoded)

**Industry Usage**:
- Text classification
- Image classification
- Bioinformatics

**Research Backing**:
- Cortes & Vapnik (1995): "Support-vector networks"
- Platt (1999): "Probabilistic outputs for SVMs"

**Configuration**:
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('svm', LinearSVC(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )),
    ('calibrator', CalibratedClassifierCV(cv=5))
])
```

---

## 3. Probabilistic Models (1 model)

### 3.1 Complement Naive Bayes (CNB)
**Algorithm**: Naive Bayes variant optimized for imbalanced classes

**How It Works**:
- Estimates probability P(y|x) using Bayes theorem
- Uses complement of each class for parameter estimation
- Assumes feature independence (naive assumption)

**Strengths**:
- Excellent for imbalanced datasets
- Fast training and inference
- Works well with sparse features
- Low memory footprint

**Weaknesses**:
- Assumes feature independence (rarely true)
- May underperform on complex patterns
- Sensitive to irrelevant features

**Best Use Cases**:
- **URL Features**: Good when features relatively independent
- **DNS Features**: Effective for DNS feature combinations
- **WHOIS Features**: Strong baseline for categorical WHOIS data

**Industry Usage**:
- Spam filtering (original use case)
- Text classification
- Real-time classification systems

**Research Backing**:
- Rennie et al. (2003): "Tackling the Poor Assumptions of Naive Bayes Text Classifiers"
- McCallum & Nigam (1998): "A comparison of event models for Naive Bayes text classification"

**Configuration**:
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('clf', ComplementNB())
])
```

---

## 4. Distance-Based Models (1 model)

### 4.1 K-Nearest Neighbors (KNN)
**Algorithm**: Instance-based learning using k nearest neighbors

**How It Works**:
- Stores all training data
- Prediction: find k nearest neighbors, majority vote
- Distance metric: Euclidean (after scaling)

**Strengths**:
- No training phase (lazy learning)
- Naturally handles multi-class
- Good for irregular decision boundaries
- Adapts to local patterns

**Weaknesses**:
- Slow inference (searches all training data)
- Memory intensive (stores all data)
- Sensitive to feature scaling and irrelevant features
- Poor on high-dimensional data (curse of dimensionality)

**Best Use Cases**:
- **URL Features**: Moderate - may struggle with high dimensionality
- **DNS Features**: Good for finding similar DNS patterns
- **WHOIS Features**: Effective for finding domains with similar registration patterns

**Industry Usage**:
- Recommendation systems (collaborative filtering)
- Anomaly detection
- Case-based reasoning systems

**Research Backing**:
- Cover & Hart (1967): "Nearest neighbor pattern classification"
- Fix & Hodges (1951): Original k-NN algorithm

**Configuration**:
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier(
        n_neighbors=adaptive_k,  # min(31, max(3, int(n_rows * 0.8 / 3)))
        weights='distance',
        metric='euclidean'
    ))
])
```

**Note**: n_neighbors is adaptive based on dataset size to prevent failures on small datasets.

---

## 5. Neural Network Models (1 model)

### 5.1 Multi-Layer Perceptron (MLP)
**Algorithm**: Feed-forward neural network with backpropagation

**How It Works**:
- Hidden layers with ReLU activation
- Backpropagation for gradient computation
- Adam optimizer for weight updates

**Strengths**:
- Learns complex non-linear patterns
- Feature learning (no manual engineering)
- Flexible architecture
- State-of-the-art on many tasks

**Weaknesses**:
- Requires more data than other methods
- Black box (less interpretable)
- Sensitive to hyperparameters
- Can overfit easily

**Best Use Cases**:
- **URL Features**: Excellent for learning URL character patterns
- **DNS Features**: Good for complex DNS feature interactions
- **WHOIS Features**: Strong on mixed categorical + numeric features

**Industry Usage**:
- Deep learning pipelines
- Complex pattern recognition
- Systems with abundant training data

**Research Backing**:
- Rumelhart et al. (1986): "Learning representations by back-propagating errors"
- LeCun et al. (2015): "Deep learning" - Nature review

**Configuration**:
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('clf', MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=500,
        random_state=42
    ))
])
```

---

## 6. Non-Linear SVM (1 model - BONUS)

### 6.1 SVM with RBF Kernel
**Algorithm**: Support Vector Machine with Radial Basis Function kernel

**How It Works**:
- Maps data to high-dimensional space via RBF kernel
- Finds maximum-margin hyperplane in kernel space
- Calibrated for probability estimates

**Strengths**:
- Handles non-linear patterns
- Effective in high-dimensional spaces
- Good generalization with proper regularization

**Weaknesses**:
- Very slow on large datasets (O(n²) to O(n³))
- Memory intensive
- Requires careful hyperparameter tuning (C, gamma)

**Best Use Cases**:
- **URL Features**: Strong but slow - good for small datasets
- **DNS Features**: Excellent for complex DNS patterns
- **WHOIS Features**: Good but may be overkill

**Industry Usage**:
- Bioinformatics
- Image classification (before deep learning)
- Small-to-medium datasets

**Research Backing**:
- Boser et al. (1992): "A training algorithm for optimal margin classifiers"
- Schölkopf et al. (1997): "Kernel principal component analysis"

**Configuration**:
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    ))
])
```

---

## Model Diversity Analysis

### By Learning Paradigm:
1. **Bagging** (2): Random Forest, Extra Trees
2. **Boosting** (5): GB, HistGB, XGBoost, LightGBM, CatBoost
3. **Linear** (4): LogReg L2, LogReg ElasticNet, SGD, Linear SVM
4. **Probabilistic** (1): Complement Naive Bayes
5. **Instance-based** (1): KNN
6. **Neural** (1): MLP
7. **Kernel** (1): SVM RBF

### By Speed (Inference Time):
1. **Fast** (<1ms): Linear models (4), CNB
2. **Medium** (1-10ms): Tree ensembles (7)
3. **Slow** (>10ms): KNN, MLP, SVM RBF

### By Interpretability:
1. **High**: Linear models (4), Decision trees
2. **Medium**: Tree ensembles (feature importance)
3. **Low**: MLP, SVM RBF, KNN

### Why 15 Models?
This diverse set creates an effective ensemble because:
- **Different inductive biases**: Each model makes different assumptions
- **Error diversity**: Models make different mistakes
- **Speed-accuracy trade-off**: Mix of fast/accurate models
- **Robustness**: Ensemble vote reduces individual model weaknesses
- **Coverage**: Different models excel on different data patterns

---

## Ensemble Strategy

### Voting Mechanism:
- **Hard voting**: Each model votes for a class (majority wins)
- **Soft voting**: Average predicted probabilities (used in production)

### Expected Performance:
- **Individual models**: 85-95% accuracy
- **Ensemble**: 95-98% accuracy (empirical testing needed)

### Latency Considerations:
- **Parallel inference**: All models run simultaneously
- **Total latency**: Max individual model latency (not sum)
- **Target**: <100ms for real-time API response

---

## Next Steps

1. **Ensemble Finding**: Test combinations to find optimal subset
2. **Deployment**: Design parallel inference architecture
3. **Latency Testing**: Empirically measure inference times
4. **A/B Testing**: Compare different ensemble configurations
5. **Monitoring**: Track individual model contributions over time

---

*Document Version: 1.0*
*Last Updated: 2025-12-30*
*Author: PhishNet ML Team*
