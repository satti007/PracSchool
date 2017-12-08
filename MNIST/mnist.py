import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten

nb_classes = 10
np.random.seed(2017) # for reproducibility

def plot_model_history(model_history,file_name):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig(file_name)

def dataPrep(conv):
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("X_train original shape", X_train.shape)
    print("y_train original shape", y_train.shape)
    
    if conv:
            X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
            X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    else:
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    # One-hot encoding the output i.e 1-->(1,0,0,0,0,0,0,0,0,0),2-->(0,1,0,0,0,0,0,0,0,0),....
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    print("Training matrix shape", X_train.shape)
    print("Training target matrix shape", Y_train.shape)
    print("Testing matrix shape", X_test.shape)
    print("Testing target matrix shape", Y_test.shape)
    
    return X_train,Y_train,X_test,Y_test

def train(X_train,Y_train,X_test,Y_test,drop_prob,activ_fun,num_epochs,callback):
    model = Sequential()
    model.add(Dense(512, input_shape=(784,))) # Hidden layer_1 with 512 neurons
    model.add(Activation(activ_fun)) 
    if drop_prob != 0:
    	model.add(Dropout(drop_prob))         # Dropout regularization to avoid overfitting
    model.add(Dense(512))                     # Hidden layer_2 with 512 neurons 
    model.add(Activation(activ_fun))
    if drop_prob != 0:
    	model.add(Dropout(drop_prob)) 
    model.add(Dense(10))                      # Output Layer with 10 neurons 
    model.add(Activation('softmax')) 
    
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    if callback:
        earlyStopping=EarlyStopping(monitor='loss', min_delta=0.0001, patience=5, verbose=0, mode='auto')
        model_info = model.fit(X_train, Y_train,callbacks=[earlyStopping],batch_size=128, 
                               epochs=num_epochs,verbose=2,validation_split=0.15)
    else:
        model_info = model.fit(X_train, Y_train,batch_size=128,epochs=num_epochs,verbose=2,validation_split=0.15)
    score = model.evaluate(X_test, Y_test,verbose=0)
    return model_info,score

def conv_train(X_train,Y_train,X_test,Y_test,num_epochs,callback):
	model = Sequential()
	model.add(Convolution2D(20, 5, strides=(1, 1), activation='relu', input_shape=(1,28,28), data_format='channels_first'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1), data_format='channels_first'))
	model.add(Convolution2D(50, 5, strides=(1, 1), activation='relu',data_format='channels_first'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1), data_format='channels_first'))
	model.add(Flatten())
	model.add(Dense(500, activation='relu'))
	model.add(Dense(10, activation='softmax'))
	 
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	if callback:
		earlyStopping=EarlyStopping(monitor='loss', min_delta=0.0001, patience=5, verbose=0, mode='auto')
		model_info = model.fit(X_train, Y_train,callbacks=[earlyStopping],batch_size=32, 
												epochs=num_epochs,verbose=2,validation_split=0.15)
	else:
	    model_info = model.fit(X_train, Y_train,batch_size=32,epochs=num_epochs,verbose=2,validation_split=0.15)
	score = model.evaluate(X_test, Y_test,verbose=0)
	return model_info,score

conv = False
X_train,Y_train,X_test,Y_test = dataPrep(conv)

'''callback = False
drop_prob = 0
activ_fun = 'relu'
num_epochs = 10
if conv:
    print 'No model'
else:
    model_info,score = train(X_train,Y_train,X_test,Y_test,drop_prob,activ_fun,num_epochs,callback)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
'''

# Varying dropout --[0.1,0.2,0.4]
probs = [0.1,0.2,0.4]
acc_1 = []
min_loss_1 = []
for p in probs:
    print 'Dropout prob: '+str(p)
    model_info,score = train(X_train,Y_train,X_test,Y_test,p,'relu',10,False)
    print('Test accuracy:', score[1])
    acc_1.append(score[1])
    fig = plt.figure()
    loss = model_info.history['loss']
    min_loss_1.append(min(loss))
    plt.plot(loss)
    plt.ylabel('train_loss')
    plt.xlabel('epochs')    
    plt.title('For Drop out = '+str(p))
    fig.savefig('drop_'+str(p)+ '.png')
    print '##########################'
print acc_1,min_loss_1
print ''

# Activation fuctions -- [sigmoid,relu,tanh]
functions = ['sigmoid','relu','tanh']
acc_2 = []
min_loss_2 = []
for f in functions:
    print 'Activation function: '+f
    model_info,score = train(X_train,Y_train,X_test,Y_test,0.2,f,10,False)
    print('Test accuracy:', score[1])
    acc_2.append(score[1])
    fig = plt.figure()
    loss = model_info.history['loss']
    min_loss_2.append(min(loss))
    plt.plot(loss)
    plt.ylabel('train_loss')	
    plt.xlabel('epochs')    
    plt.title('For Activation function: '+f)
    fig.savefig(f +'.png')
    print '##########################'
print acc_2, min_loss_2
print ''

# No.of epochs
print 'Training for 100 epochs'
model_info,score = train(X_train,Y_train,X_test,Y_test,0.2,'relu',100,False)
print('Test accuracy:', score[1])
plot_model_history(model_info,'epochs_100.png')
print '##########################'
print ''

print 'Training for 100 epochs with earlystopping'
model_info,score = train(X_train,Y_train,X_test,Y_test,0.2,'relu',100,True) # earlystopping
print('Test accuracy:', score[1])
plot_model_history(model_info,'earlystopping.png')
print '##########################'
print ''


## Lenet-5 Model
conv = True
X_train,Y_train,X_test,Y_test = dataPrep(conv)

print 'Training Lenet Model for 100 epochs'
model_info,score = conv_train(X_train,Y_train,X_test,Y_test,100,False) 
print('Test accuracy:', score[1])
plot_model_history(model_info,'lenet_100.png')
print '##########################'
print ''

print 'Training Lenet Model for 100 epochs with earlystopping'
model_info,score = conv_train(X_train,Y_train,X_test,Y_test,100,True) # earlystopping
print('Test accuracy:', score[1])
plot_model_history(model_info,'lenet_earlystopping.png')
print '##########################'
print ''
















'''
import re

file = open('train.log','r')
data = file.read()
loss = re.findall(r' loss: (.*?) ',data,re.DOTALL)
acc = re.findall(r' acc: (.*?) ',data,re.DOTALL)
val_loss = re.findall(r' - val_loss: (.*?) - val_acc: ',data,re.DOTALL)
val_acc = re.findall(r' - val_acc: (.*?)\n',data,re.DOTALL)
'''