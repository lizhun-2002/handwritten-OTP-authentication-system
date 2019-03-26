# [Handwritten One-time Password Authentication System Based On Deep Learning](http://kiss.kstudy.com/thesis/thesis-view.asp?key=3660839)
*Original paper: [pdf](./paper/HandwrittenOPT_ZhunLi.pdf)*

This is a demostration of handwritten one-time password authentification system on Python 3, Keras, and TensorFlow. 
The original paper has been published in JICS journal.

We propose a handwritten one-time password authentication system which employs deep learning-based handwriting recognition and writer verification techniques. We design a convolutional neural network to recognize handwritten digits and a Siamese network to compute the similarity between the input handwriting and the genuine user’s handwriting.

## Requirements
Python 3.5.5, TensorFlow 1.2.1, and Keras 2.0.6

## Data
We proposed the first known application of the second edition of NIST Special Database 19 (SD19) in a writer verification task.
Download the data from [here](https://www.nist.gov/srd/nist-special-database-19)

## Handwriting recognition
To train the model, use
```python
python train_cnn_recognition.py
```
To test the model, use
```python
python test_cnn.py
```

## Writer verification
To train the model, use
```python
python train_siamese_v3.py
```
To test the model, use
```python
python test_siamese_v1.py
```
