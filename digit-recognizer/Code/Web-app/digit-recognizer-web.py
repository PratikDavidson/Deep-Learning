import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from PIL import Image
import tensorflow as tf

header = st.container()
data_desc = st.container()
model_desc = st.container()
results = st.container()


@st.cache
def load_data():
    data = pd.read_csv('D:\\ML\\DL Repo\\digit-recognizer\\data\\train.csv')
    return data


def image_show(img, class_label):
    fig = plt.figure()
    plt.title('Digit-' + str(class_label))
    plt.imshow(img, cmap=plt.cm.gray)
    st.pyplot(fig)


@st.cache
def model_data():
    fcnn = keras.models.load_model('D:\\ML\DL Repo\\digit-recognizer\\model\\Saved Models\\best_FCNN_model.h5')
    _1d_cnn = keras.models.load_model('D:\\ML\DL Repo\\digit-recognizer\\model\\Saved Models\\best_1D_CNN_model.h5')
    _2d_cnn = keras.models.load_model('D:\\ML\DL Repo\\digit-recognizer\\model\\Saved Models\\best_2D_CNN_model.h5')
    return fcnn, _1d_cnn, _2d_cnn


@st.cache
def load_image_plots():
    image_plots_fcnn = Image.open('D:\\ML\\DL Repo\\digit-recognizer\\model\\Saved Plots\\model_plot_FCNN.png')
    image_plots_1d_cnn = Image.open('D:\\ML\\DL Repo\\digit-recognizer\\model\\Saved Plots\\model_plot_1D_CNN.png')
    image_plots_2d_cnn = Image.open('D:\\ML\\DL Repo\\digit-recognizer\\model\\Saved Plots\\model_plot_2D_CNN.png')
    # res_FCNN_image_plots = FCNN_image_plots.resize((500,500))
    # res_1D_CNN_image_plots = _1D_CNN_image_plots.resize((500, 500))
    # res_2D_CNN_image_plots = _2D_CNN_image_plots.resize((500, 500))
    return image_plots_fcnn, image_plots_1d_cnn, image_plots_2d_cnn


def model_summary(model_type):
    with open('D:\\ML\\DL Repo\\digit-recognizer\\model\\Saved summary\\' + model_type + '_model_summary.txt') as f:
        st.markdown(f.read())


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

def model_history(model_type):
    history = pd.read_csv(
        'D:\\ML\DL Repo\\digit-recognizer\\model\\Saved training data\\' + model_type + '_model_history.csv')
    best_val_loss = np.argmin(np.array(history['val_loss']))
    return history, best_val_loss

with st.sidebar:
    photo = st.image(Image.open('D:\\ML\\DL Repo\\me.jpg').resize((300, 300)))
    with open('D:\\ML\\DL Repo\\about_me.md') as f:
        st.markdown(f.read())


with header:
    st.title('Kaggle - Digit Recognizer')
    st.header('Description')
    st.write(
        'Digit Recognizer is based on identifying handwritten digits from the "Hello World!" dataset in computer '
        'vision (MNIST).',
        align_text='center')

with data_desc:
    train_data = load_data()
    st.header('Dataset')
    st.write(
        'The dataset used in this project can be downloaded from [here]('
        'https://www.kaggle.com/competitions/digit-recognizer/data).',
        align_text='center')
    st.write(
        'The data files train.csv and test.csv contain 42000 and 28000 gray-scale images of hand-drawn digits '
        'respectively, from zero through nine.',
        align_text='center')
    st.write(
        'Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has '
        'a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher '
        'numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive. The training data set, '
        '(train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The '
        'rest of the columns contain the pixel-values of the associated image.',
        align_text='center')
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
        best_acc = round(history['val_accuracy'].iloc[best_val_loss_idx]*100, 2)
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
    st.write(
        '2D-CNN performed far better in comparison to other two models with least learning parameters whereas 1D-CNN '
        'though performed a little bit better than FCNN but the overall advantage was less learning parameters '
        'compared to FCNN.',
        align_text='center')
