import os
import datetime
import numpy as np
from nltk import FreqDist
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense,  Activation

def load_data(source, max_len, vocab_size):
    # Reading raw text from source and destination files
    f = open(source, 'r')
    X_data = f.read()
    f.close()
    print "Length of X_data: ",len(X_data)
    
    # Splitting raw text into array of sequences and truncating the seq if (len of seq > max_len)
    X = [text_to_word_sequence(x) for x in X_data.split('\n') if len(x) > 0]
    
    X_max_len = max([len(sentence) for sentence in X])
    print "Max Length of a review before: ",X_max_len
    for index,sequence in enumerate(X):
        if len(sequence) > max_len:
            X[index] = sequence[:max_len]
    
    X_max_len = max([len(sentence) for sentence in X])
    print "Max Length of a review after: ",X_max_len
    
    # Creating the vocabulary set with the most common words
    dist = FreqDist(np.hstack(X))
    X_vocab = dist.most_common(vocab_size-1)
    print "Length of X_vocab is: ",len(X_vocab)
    
    # Creating an array of words from the vocabulary set, we will use this array as index-to-word dictionary
    X_ix_to_word = [word[0] for word in X_vocab]
    # Adding the word "ZERO" to the beginning of the array
    X_ix_to_word.insert(0, 'ZERO')
    # Adding the word 'UNK' to the end of the array (stands for UNKNOWN words)
    X_ix_to_word.append('UNK')
    
    # Creating the word-to-index dictionary from the array created above
    X_word_to_ix = {word:ix for ix, word in enumerate(X_ix_to_word)}
    
    # Converting each word to its index value
    for i, sentence in enumerate(X):
        for j, word in enumerate(sentence):
            if word in X_word_to_ix:
                X[i][j] = X_word_to_ix[word]
            else:
                X[i][j] = X_word_to_ix['UNK']
    
    return (np.array(X), len(X_vocab)+2, X_word_to_ix, X_ix_to_word)

def divide_data(X):
    X_pos = X[12500:]
    X_neg = X[0:12500]
    np.random.shuffle(X_pos)
    np.random.shuffle(X_neg)
    
    X_pos_train,X_pos_test = X_pos[0:7500], X_pos[7500:]
    X_neg_train,X_neg_test = X_neg[0:7500], X_neg[7500:]
    
    X_test = np.concatenate((X_neg_test,X_pos_test))
    X_train = np.concatenate((X_neg_train,X_pos_train))
    np.random.shuffle(X_train)
    np.random.shuffle(X_test)
    
    return X_train,X_test


def buid_model(X_vocab_len,latent_dim):
    encoder_inputs = Input(shape=(None, X_vocab_len))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]    # We discard `encoder_outputs` and only keep the states.
    
    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, X_vocab_len))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)
    decoder_dense = Dense(X_vocab_len, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model that will turn `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
    print model.summary()
    
    return model

def process_data(word_sentences, max_len, word_to_ix):
    # Vectorizing each element in each sequence
    sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
    target_sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
    for i, sentence in enumerate(word_sentences):
        for j, word in enumerate(sentence):
            sequences[i, j, word] = 1
            if j > 0:
                target_sequences[i, j-1, word] = 1
    return sequences, target_sequences


num_epochs = 20
max_len = 200
vocab_size = 3000
latent_dim = 1000
batch_size = 10
Mode = 'train'

print('Loading data...')
X, X_vocab_len, X_word_to_ix, X_ix_to_word = load_data('merged_train.txt',max_len,vocab_size)

# Padding zeros to make all sequences have a same length with the longest one
print('[INFO] Zero padding...')
X = pad_sequences(X, maxlen=max_len, dtype='int32')

# Dividing data into 15K train & 10K test
print('[INFO] Dividing data...')
X_train, X_test = divide_data(X)
print('[INFO] Training data size: ',len(X_train))
print('[INFO] Test data size: ',len(X_test))

model = buid_model(X_vocab_len,latent_dim)

start = 0
if Mode == 'train':
    for k in range(1, num_epochs+1):
        np.random.shuffle(X_train)
        
        # Training 100 sequences at a time
        for i in range(0, len(X_train), 100):
            if i + 100 >= len(X_train):
                i_end = len(X_train)
            else:
                i_end = i + 100
            
            encoder_input_data,decoder_target_data = process_data(X_train[i:i_end], max_len, X_word_to_ix)
            print('[INFO] Training model: epoch {}th {}/{} samples'.format(k, i, len(X_train)))
            if start:
                model.load_weights('load_checkpoint.hdf5')
                print "Loaded weights!!"
            model.fit([encoder_input_data, encoder_input_data], decoder_target_data,batch_size=batch_size,
                                                        epochs=1,validation_split=0.2,verbose=2)
            model.save_weights('load_checkpoint.hdf5')
            start = 1
        
        model.save_weights('checkpoint_epoch_{}.hdf5'.format(k))
