# Facial Expression Recognition with Identity Disentanglement
This project focuses on disentangling expression and identity features for improved facial expression recognition using deep learning.

It uses two encoders:

Expression Encoder (Eexp): Captures facial expressions

Identity Encoder (Eid): Captures identity-specific features

These encoders are used to train a classifier that distinguishes between expressions using a hybrid feature vector.
##  Dataset Collection
A webcam-based Python script captures face images labeled by expression.

Press keys (n, a, h, etc.) to label images as Neutral, Angry, Happy, etc.

Images are cropped to 250×250 and saved in folders by label.
## Model Components
xpressionEncoder: Learns expression features without identity interference.

IdentityEncoder: Learns identity-specific traits to help isolate expression information.

Classifier: Learns to classify emotions using concatenated features from both encoders.
## Training
The train_expression_classifier function:

Freezes both encoders.

Trains a classifier on top of their combined output (expression + identity features).

Uses cross-entropy loss and tracks accuracy.
##  Evaluation
The evaluate_classifier function:

Tests the trained classifier on unseen data.

Reports final test accuracy and returns predicted/true labels.
## Requirements
Python 3.8+, PyTorch, OpenCV, NumPy, torch, torchvision, Pillow, numpy, matplotlib, scikit-learn, seaborn
