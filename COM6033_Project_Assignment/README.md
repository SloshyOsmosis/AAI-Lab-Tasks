# UK Bird Prediction App

This project is a web app built in python for predicting the species of a bird from an uploaded image. The application supports the classification of 5 common species of birds than can be found in the UK: GoldFinches, Magpie's, Robin's, Sparrow's, and Swan's.

The application uses a pre-trained Convolutional Neural Network (CNN) model for multiclass classification to predict the bird species.

The dataset used for training the model contains roughly 1000 images for each class of bird, totalling in 5248 images of birds.
## Features
+ Upload an image of a bird.
+ Predicts one of the following species: Goldfinch, Magpie, Robin, Sparrow, and Swan.
+ Displays the top 3 predictions with confidence percentages.
+ Displays the uploaded image.

## Prerequisites 
### Flask Deployment Dependencies
+ Python (3.9.18)
+ Flask (3.1.0)
+ Tensorflow/Keras (2.18.0)
+ Pillow (11.0.0)
+ NumPy (2.0.2)

### Anaconda JupyterLab Dependencies (For the project_assignment.ipynb notebook)
+ Python (3.9.18)
+ Tensorflow (2.10.0)
+ Keras (2.10.0)
+ NumPy (2.0.2)
+ Matplotlib (3.9.2)

## Installation
To ensure a clean and isolated workspace you need to create a python virtual environment'

1. Create a Virtual environment in the powershell terminal
+ python -m venv .venv

2. Activate the virtual environment after creation
+ .venv/Scripts/activate
+ Once successful, you should see '(venv)' in the command line

3. Install the required libaries
+ pip install -r requirements.txt

4. Run the flask application
+ python app.py
OR
+ flask run

5. Once step 1 through 4 have been completed, access the application in your browser from this url: http://127.0.0.1:5000/

## How to use
1. Upload an image of a bird from the list of 5

2. Click 'Upload and predict'

3. See results!
## Technologies Used
### Backend
+ Python
+ Flask
+ TensorFlow/Keras
+ Pillow
+ Numpy
+ bing_image_installer (for dataset collection)
### Frontend
+ HTML/CSS
+ Base64

## Future Improvements and Challenges faced
In the future if I were to undertake this project again, I would include more species of birds commonly found in the UK such as feral pigeons, crows, wrens, etc. Doing this would improve the models real world applicability.

I would also further increase the size of the dataset. During training iterations, the model had trouble distinguishing features between the 'sparrow' and 'robin'. A larger dataset would give the model more features to work with and have an easier time distinguishing these two classes apart.

A challenge I was faced with early during development was an imbalanced dataset. While the website images.cv was able to provide most of the images, a few classes such as the 'sparrow' and 'goldfinch' classes we're severely unbalanced, only containing 600 images each. Rather than downloading images from google manually, I took advantage of the python library bing_image_downloader to pad out the dataset with new data. This method was not perfect as it required quite a lot of cleaning by manually deleting images of tattoos, ai generated images, and illustrations of birds. During development, the imbalanced classes caused the model to become more biased towards the other classes during training.

## Reflections
The journey of building this application has been both challenging and rewarding. The time spent on this model allowed me to develop a greater understanding of how convolutional neural network handles image data by fine tuning the model for accuracy and efficiency. The challenges i was faced during development were seen as learning opportunities.

My management of the imbalanced datasets was seen as an opportunity to introduce data augmentation to my model. 
## References

Image datasets (Covered half of the dataset) - https://images.cv/search-labeled-image-dataset
Image collection python library (Padding out the dataset for the smaller classes) - bing_image_downloader