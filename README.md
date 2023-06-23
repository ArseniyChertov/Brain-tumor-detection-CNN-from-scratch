# Brain-tumor-detection-CNN-from-scratch
-The goal is to determine whether or not an MRI contains a tumor.(binary classification)
-The input images varied by size so I resized them to (512, 512).Therefore some information was lost. I did not use any interpolation techniques to preserve the quality.
-I normalized the pixel values between 0 and 1 to scale.
- The dataset was also very limited and contained 253 total samples(both with and without tumors) so I decided to split it into 20 test, 20 validation, and 213 training samples. Obviously more samples should have been used for testing and validation.
-This model is still being optimized and is very small considering the size of the dataset.
-The model consists of a convolutional layer, max pooling, and 2 fully connected layers. Sigmoid was used in the second fully connected layer since binary classification. ‘he_uniform kernel initializer seemed to have the best initialization.




Dataset can be found on Kaggle. Manual splitting must be done.
