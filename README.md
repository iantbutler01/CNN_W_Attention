# Instructions for use.
Use only with python >= 3.6 

## Instructions assume use of Python3
* Clone the repository
* (Optional) create a python >= 3.6 virtual env
* run "pip install -r requirements.txt"
* run "python input_processing.py Train"
  * This will produce a 'dictionaries.p' and a train_dataset.p file
* run "python cnn.py" or "cnn_w_attention.py" or "cnn_w_attention_fasttext.py" to train and save the model you are interested in.

## For the API

* cd api
* 'python api.py
* http://localhost:5000 there will be instructions on what to do
*
