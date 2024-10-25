import streamlit as st 
import torch 
from torch.nn import functional as F
from text_preprocessor import Text_preprocessor
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import numpy as np 

PATH='sentiment_lstm_cpu.pt'

def show_piechart(values):
    labels = ['Negative', 'Positive']
    sizes = values.squeeze()
    explode = (0, 0.1) 

    # Adjust figure size and font size
    fig1, ax1 = plt.subplots(figsize=(4, 4))  # Decreased figure size
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'color': 'white', 'fontsize': 7})  # Decreased font size
    
    ax1.axis('equal') 
    fig1.patch.set_alpha(0)  
    fig1.patch.set_facecolor('#333333') 
    st.pyplot(fig1, use_container_width=True)
    




def load_model(path):
    model = torch.jit.load(path)
    model.eval()
    return model

def predict(model, text, preprocessor):
    labels = ['Negative', 'Positive']
    des_vec, length = preprocessor.description_to_vector(text)
    padded_res = preprocessor.vector_padding(des_vec).squeeze()
    padded_res = padded_res.unsqueeze(0)
    res = model(padded_res)
    percentage = F.softmax(res, dim=1)
    forward = torch.argmax(F.softmax(res, dim=1), dim=1)
    return labels[forward], percentage
    
        
def main():
    st.title('Sentiment Analysis using lstm')
    model = load_model(PATH)
    preprocessor = Text_preprocessor()
    preprocessor.max_length = 240
    text = st.text_input('Enter the sentence to find the sentiment: ')

    if text:
        with torch.no_grad():
            try:
                pred, percent = predict(model, text, preprocessor)
                color = 'yellow' if pred == 'Positive' else 'blue'
                st.markdown(f'### <span style="color: {color};">{pred}</span>', unsafe_allow_html=True)
                show_piechart(percent)

            except RuntimeError: 
                print('Enter Longer text')


    

    

if __name__ == '__main__':
    main()

