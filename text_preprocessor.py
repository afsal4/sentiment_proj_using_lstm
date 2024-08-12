import torch 
import re 
import spacy
import numpy as np 
from spacy.cli import download

download("en_core_web_md")


class Text_preprocessor():
    
    
    def __init__(self):
        self.max_length = 0
        self.train_files = []
        self.test_files = []
        self.nlp = spacy.load('en_core_web_md')
        
    def preprocess(self, X_train, y_train, train=True):
        '''
        X_train = text or sentence of the review
        y_train = labels in vectorized format
        '''
        
        self.data_to_vector(X_train, y_train, train) # converts the data to vector and removes the data if the sent is zero and save it in a file
        files = self.train_files if train else self.test_files
        if train: # if its train mode view the distribution and fix the pad size else do the padding 
            return self.plot_length(self.train_files) 
        else:
            test_batches = self.data_vec_to_pad(self.test_files) # padd the sentence with zeros so the every rows the same returns the vec as batches

            return test_batches
        
    
    def description_to_vector(self, string):
        string = re.sub('.*?>', '', string)
        tokenizer = self.nlp(string)
        words = []

        for token in tokenizer:

            if token.is_punct or token.is_stop:
                continue

            if token.has_vector:
                words.append(torch.from_numpy(token.vector))

        if len(words) == 0:
            return [], 0
    
        vectorized = torch.stack(words) 
        return vectorized, len(vectorized)


    def vector_padding(self, arr):
        # the arr rows are words and col is word embeddings of that word 
        # to do add zeros rows so that rows equal to max_size
        row_length, col_length = arr.shape # if the shape is of row and col
        if row_length > self.max_length:
            padded = arr[:self.max_length, :]
        else:
            zero_row_length = self.max_length - row_length
            padded = torch.cat((arr, torch.zeros(zero_row_length, col_length)))
        return padded
    


    def data_to_vector(self, X_train, y_train, train=True):


        vectors = []
        labels = []
        files = []
        file_no = 0


        for idx, i in enumerate(X_train):
            vector, vector_len = self.description_to_vector(i)
            if vector_len > 0:
#                 self.max_length = max(self.max_length, vector_len) 

                vectors.append(vector)
                labels.append(y_train.iloc[idx])

            if (idx+1) % 500 == 0:
                # saves files in 5000 iterations 
                file_no = idx / 5000
                if train:
                    file_no = f'Train_{file_no}'
                vecno = f'/kaggle/working/Vector_{file_no}.pt'

                vectors, labels = vectors, torch.tensor(labels)
                torch.save((vectors, labels), vecno)
                if train:
                    self.train_files.append(vecno)
                else:
                    self.test_files.append(vecno)
                vectors = []
                labels = []


                print(f'----{idx}----')

        if vectors:
            vecno = f'/kaggle/working/Vector_{file_no}_final.pt'

            vectors, labels = vectors, torch.tensor(labels)
            torch.save((vectors, labels), vecno)
            files.append(vecno)
            vectors = []
            labels = []
        print('Finished Vectorizing')
    


    def data_vec_to_pad(self, files):
        x_values = []

        for file in files:
            X, y = torch.load(file)
            for i in range(len(X)):
                X[i] = self.vector_padding(X[i])
            x_values.append((torch.stack(X), y)) # (8, (tuple of total x, total y))

        return x_values


    def plot_length(self, files):
        lengths = []
        print(files)
        for file in files:
            (X, y) = torch.load(file)
            for i in X:
                lengths.append(len(i))
        lengths = np.array(lengths)
        return lengths

