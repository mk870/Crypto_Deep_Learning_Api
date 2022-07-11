# Crypto Deep Learning Predictive AI
<img src="https://i.ibb.co/VxNMSCs/lstm.png" alt="lstm" border="0"> 

## Project Summary 
* This application takes in 60 historic prices (daily) of bitcoin, ethereum and litecoin and returns predicted future prices. 
* Used Historical price data from yahoo finance using Pandas data Reader.
* Collected Bitcoin data from 2014, Ethereum from 2017, Litecoin from 2014.
* Prepared the data for the deep learning model.
* Built Bidirectional LTSM model to predict the prices.
* Used google colab to train the model so as to utilize free GPUs.
* Built a client facing Api using flask and hosted it on Heroku  
## The following is the Api in action:

<img src="https://i.ibb.co/72WKLfj/Crypto-Mania.png" alt="Crypto-Mania" border="0"> 

### **Resources Used**
***
**Python Version**: 3.8

**Packages**: Pandas, Numpy, Sklearn, Json, Flask, Pickle, Yahoo finace Api, Jupyter notebook, Tensorflow and Keras.  
![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=flat&logo=flask&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white) ![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=flat&logo=googledrive&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white) ![Bitcoin](https://img.shields.io/badge/Bitcoin-000?style=flat&logo=bitcoin&logoColor=white) ![Ethereum](https://img.shields.io/badge/Ethereum-3C3C3D?style=flat&logo=Ethereum&logoColor=white) ![Litecoin](https://img.shields.io/badge/Litecoin-A6A9AA?style=flat&logo=Litecoin&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=flat&logo=Keras&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)

**For Web Framework Requirements**: pip install -r requirements.txt

**APIs**: Yahoo finance API

### **Data Collection**
***
Used Pandas Data Reader to pull data from Yahoo finance into a dataframe. Pulled historical Bitcoin, Etherum and Litecoin from data 2014, 2017 and 2014 respectively.

### **Data Cleaning**
After data collection, data cleaning commenced with the following steps: 
* Split the data into training and test sets at 70:30 ratio using the Sklearn library.
* Created a function that shapes and transforms the dataset into clear features and targets, with features being the previous 60 prices and the target being the 61st price respectively.
* Scaled the data using min-max scaler package from sklearn.
* Reshaped the data into a 3d tensor, a prerequisite for LSTM models.


### **Model Building**
***

* Built a sequential model from tensorflow.
* Used the bidirectional LSTM model with 3 layers for each crypto coin. The models' architecture comprise of an input layer with 128 neurons, using the relu activation function, and a neuron dropout rate of 50%.  
The second layer has 64 neurons, uses the relu activation function, and a neuron dropout rate of 50%.  
The final layer has a dense layer with 1 neuron for the final predictions. 
* Trained all 3 models and tested using the test data leading to the following results:  

### Bitcoin model : **RMSE** 2706.98 , **R-squared score** 97.99%  
 **Actual vs Predicted Historical Prices Graph**  
 <img src="https://i.ibb.co/8gg8Sp0/Crypto-Deep-Learning-Api-Model-Building-ipynb-at-main-mk870-Crypto-Deep-Learning-Api.png" alt="Crypto-Deep-Learning-Api-Model-Building-ipynb-at-main-mk870-Crypto-Deep-Learning-Api" border="0">  

### Ethereum model : **RMSE** 260.55 , **R-squared score** 92.25%  
**Actual vs Predicted Historical Prices Graph**  
<img src="https://i.ibb.co/vvFRpNB/Crypto-Deep-Learning-Api-Model-Building-ipynb-at-main-mk870-Crypto-Deep-Learning-Api-1.png" alt="Crypto-Deep-Learning-Api-Model-Building-ipynb-at-main-mk870-Crypto-Deep-Learning-Api-1" border="0">  

### Litecoin model : **RMSE** 12.84 , **R-squared score** 96.72%  
**Actual vs Predicted Historical Prices Graph**  
<img src="https://i.ibb.co/LpKfSYc/Crypto-Deep-Learning-Api-Model-Building-ipynb-at-main-mk870-Crypto-Deep-Learning-Api-2.png" alt="Crypto-Deep-Learning-Api-Model-Building-ipynb-at-main-mk870-Crypto-Deep-Learning-Api-2" border="0"> 

### **Productionization**
***

In this step, I built a flask API endpoint thats hosted on Heroku. I did the following:
* Created a flask backend application.
* Created 3 Get endpoints for each coin with its own model which takes in a query (60 subsequent historical prices and the number of days in the future to predict) from the client.
* The endpoints return the predicted prices as a list (array).

**Live Implemantation:** [CryptoMania]([react-cryptomania.herokuapp.com](https://react-cryptomania.herokuapp.com/))

