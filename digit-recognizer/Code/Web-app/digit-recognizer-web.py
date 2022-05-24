import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from PIL import Image
from google.oauth2 import service_account
from google.cloud import storage
import io
import tensorflow as tf

st.set_page_config(page_title='Deep Learning - Digit Recognizer')

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)
bucket_name = "digit-recognizer-data"


# @st.experimental_memo(ttl=600)
def read_file(file_path, flag=False):
    bucket = client.bucket(bucket_name)
    if flag:
        content = bucket.blob(file_path).download_as_bytes().decode('utf-8')
    else:
        content = bucket.blob(file_path).download_as_bytes()
    return content


@st.experimental_memo()
def load_data():
    data = pd.read_csv(io.BytesIO(read_file('data/train.csv')), sep=',')
    return data


def image_show(img, class_label):
    fig = plt.figure()
    plt.title('Digit-' + str(class_label))
    plt.imshow(img, cmap=plt.cm.gray)
    st.pyplot(fig)


@st.experimental_memo()
def model_data():
    fcnn = keras.models.load_model(io.BytesIO(read_file('model/Saved models/best_FCNN_model.h5')))
    _1d_cnn = keras.models.load_model(io.BytesIO(read_file('model/Saved models/best_1D_CNN_model.h5')))
    _2d_cnn = keras.models.load_model(io.BytesIO(read_file('model/Saved models/best_2D_CNN_model.h5')))
    return fcnn, _1d_cnn, _2d_cnn


@st.experimental_memo()
def load_image_plots():
    image_plots_fcnn = Image.open(io.BytesIO(read_file('model/Saved plots/model_plot_FCNN.png')))
    image_plots_1d_cnn = Image.open(io.BytesIO(read_file('model/Saved plots/model_plot_1D_CNN.png')))
    image_plots_2d_cnn = Image.open(io.BytesIO(read_file('model/Saved plots/model_plot_2D_CNN.png')))
    # res_FCNN_image_plots = FCNN_image_plots.resize((500,500))
    # res_1D_CNN_image_plots = _1D_CNN_image_plots.resize((500, 500))
    # res_2D_CNN_image_plots = _2D_CNN_image_plots.resize((500, 500))
    return image_plots_fcnn, image_plots_1d_cnn, image_plots_2d_cnn


def model_summary(model_type):
    st.markdown(read_file('model/Saved summary/' + model_type + '_model_summary.txt', flag=True))


def plot(history):
    metric = "accuracy"
    fig = plt.figure()
    plt.plot(history[metric])
    plt.plot(history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    return fig


@st.experimental_memo()
def model_history(model_type):
    history = pd.read_csv(
        io.BytesIO(read_file('model/Saved training data/' + model_type + '_model_history.csv')), sep=',')
    best_val_loss = np.argmin(np.array(history['val_loss']))
    return history, best_val_loss


st.markdown(f'<style>{read_file("style.css", flag=True)}</style>', unsafe_allow_html=True)

with st.sidebar:
    photo = st.image(Image.open(io.BytesIO(read_file('me.png'))).resize((300, 300)))
    col_1, col_2, col_3 = st.columns([1.5,5,1])
    with col_1:
        pass
    with col_2:
        st.markdown('# Pratik Davidson')
    with col_3:
        pass
    st.info(read_file('about_me.md', flag=True))

header = st.container()
data_desc = st.container()
model_desc = st.container()
results = st.container()

with header:
    st.title('Kaggle - Digit Recognizer')
    st.header('Description')
    st.write('<div style="text-align:justify">' +
             '''Digit Recognizer is based on identifying handwritten digits from the "Hello World!" dataset(MNIST) in computer 
             vision.''' + '<div>', unsafe_allow_html=True)

with data_desc:
    train_data = load_data()
    st.header('Dataset')
    st.write(
        '''The dataset was taken from [Digit Recognizer competition](
        https://www.kaggle.com/competitions/digit-recognizer/overview) on Kaggle.''',
        align_text='center')

    st.markdown('<div style="text-align:justify">' + '''<p>The training dataset consists of 42000 gray-scale images of 
                 hand-drawn digits from zero through nine respectively. The first 
                 column, named "label", represents the digit. The rest of the columns represents the pixel values of the 
                 associated image of the digits in total 784 pixels for each. These 784 pixels after reshaping can 
                 represent an image of 28 pixels in height and 28 pixels in width with each
                 pixels value ranging between 0 and 255 that is the grayscale range.
                 Below is how the train data looks like and the images of respective row data.</p>''' + '</div>',
                unsafe_allow_html=True)

    st.write(train_data.head(6))
    img_section_1 = st.container()
    img_section_2 = st.container()
    data_np = np.array(train_data[train_data.columns[1:]]).reshape((train_data.shape[0], 28, 28))
    label = np.array(train_data[train_data.columns[0]])
    with img_section_1:
        col_1, col_2, col_3 = st.columns(3)
        with col_1:
            image_show(data_np[0], label[0])
        with col_2:
            image_show(data_np[1], label[1])
        with col_3:
            image_show(data_np[2], label[2])
    with img_section_2:
        col_1, col_2, col_3 = st.columns(3)
        with col_1:
            image_show(data_np[3], label[3])
        with col_2:
            image_show(data_np[4], label[4])
        with col_3:
            image_show(data_np[5], label[5])

with model_desc:
    st.header('Model Description')
    FCNN_desc = st.container()
    _1D_CNN_desc = st.container()
    _2D_CNN_desc = st.container()
    FCNN, _1D_CNN, _2D_CNN = load_image_plots()
    # FCNN_model, _1D_CNN_model, _2D_CNN_model = model_data()
    with FCNN_desc:
        st.subheader('1. Fully-Connected Neural Network')
        st.markdown('#### Model Summary:')
        model_summary('FCNN')
        st.markdown('#### Model Results:')
        col_1, col_2 = st.columns(2)
        history, best_val_loss_idx = model_history('FCNN')
        best_acc = round(history['val_accuracy'].iloc[best_val_loss_idx] * 100, 2)
        with col_1:
            st.pyplot(plot(history))
        with col_2:
            st.metric('Accuracy', f'{best_acc}%')

    with _1D_CNN_desc:
        st.subheader('2. 1-D Convolutional Neural Network')
        st.markdown('#### Model Summary:')
        model_summary('1D_CNN')
        st.markdown('#### Model Results:')
        col_1, col_2 = st.columns(2)
        history, best_val_loss_idx = model_history('1D_CNN')
        best_acc = round(history['val_accuracy'].iloc[best_val_loss_idx] * 100, 2)
        with col_1:
            st.pyplot(plot(history))
        with col_2:
            st.metric('Accuracy', f'{best_acc}%')

    with _2D_CNN_desc:
        st.subheader('3. 2-D Convolutional Neural Network')
        st.markdown('#### Model Summary:')
        model_summary('2D_CNN')
        st.markdown('#### Model Results:')
        col_1, col_2 = st.columns(2)
        history, best_val_loss_idx = model_history('2D_CNN')
        best_acc = round(history['val_accuracy'].iloc[best_val_loss_idx] * 100, 2)
        with col_1:
            st.pyplot(plot(history))
        with col_2:
            st.metric('Accuracy', f'{best_acc}%')

with results:
    st.header('Conclusion')
    st.markdown('<div style="text-align:justify">' + '<p>' +
                '2D-CNN performed far better in comparison to other two models with least learning parameters whereas 1D-CNN '
                'though performed a little bit better than FCNN but the overall advantage was less learning parameters '
                'compared to FCNN.' + '</p>' + '</div>', unsafe_allow_html=True)
    st.write(
        '''You can find the notebook codes for the DL models from my [Github link](
        https://github.com/PratikDavidson/Deep-Learning/tree/master/digit-recognizer).''',
        align_text='center')
