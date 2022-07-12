import streamlit as st
from fastai.vision.core import PILImage
from fastai.learner import load_learner
from PIL import Image

# load the model
wildlife_predictor = load_learner('./wildlife_model.pkl')


def make_inference(img):
    # make predictions
    return wildlife_predictor.predict(PILImage.create(img))


def get_id(animal):
    if animal == 'cheetah':
        return 0
    if animal == 'lion':
        return 1
    else:
        return 2


st.title('Animal classifier')
'This app uses a neural network to classify images into three categories: cheetahs, tigers, and lions.'
'If you want to test it, just upload an image of your image of choice below.'
'Do note, that if you upload an image that is not in one of the three categories you will get some funny results, ' \
'but feel free to experiment. '
image = st.file_uploader("Upload an image")

if image is not None:
    st.image(Image.open(image))
    pred = make_inference(image)
    'Click classify to find out, whether your image is a cheetah, a tiger, or a lion.'
    if st.button('Classify'):
        'Your image displays a ', pred[0], ", with probability: ", str(pred[2][get_id(pred[0])].item()), '.'
