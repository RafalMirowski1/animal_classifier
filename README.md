# Animal clasifier app
A simple neural net based classifier that allows users to upload a picture of either a cheetah, a tiger, or a lion, and classifies the image.
The app was deployed using Streamlit and may be accessed at https://rafalmirowski1-animal-classifier-main-omf2li.streamlitapp.com/.
# How it works
The classifier uses a ResNet18 convolutional neural network that was pretrained on ImageNet dataset, but the final classification layer is replaced with a new one.
This approach to machine learning is an example of transfer learning; it allows one to utilize existing work for different, sufficiently similar problems.
