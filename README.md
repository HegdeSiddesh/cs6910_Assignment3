# CS6910 Assignment 3

# ASG3_Q1.ipynb

This python notebook helps in building an RNN based seq2seq model which supports lstm, gru and rnn cells. 

The function **create_model** takes in required parameters such as num_encoder_tokens ,embedding_size ,cell_type ,latent_dimension ,dropout ,number_of_encoder_layers ,num_decoder_tokens ,number_of_decoder_layers. It returns the seq2seq model as output.

The function **fit** takes in required parameters such as model,cell_type,encoder_input_data, decoder_input_data, decoder_target_data,batch_size, epochs,number_of_encoder_layers,number_of_decoder_layers and latent_dimension.It returns encoder_model and decoder_model.

The function **accuracy** takes in required parameters such as val_encoder_input_data, y_val,number_of_decoder_layers,target_token_index , cell_type ,reverse_target_char_index, max_decoder_seq_length, encoder_model and decoder_model. It returns the validation accuracy.

The function **decoded_sentence** is used to transliterate a word in the source language into the target language.

**Reference**:- https://keras.io/examples/nlp/lstm_seq2seq/

# ASG3_Q2_sweep_final.ipynb
This python notebook contains code to configure and run wandb sweeps.

# ASG_3_Best_model_without_attention.ipynb
This python notebook takes the parameters of the best model from the sweep.

Trains the model.

Makes predictions on test data and computes the test accuracy.

# plotPredictions.py
This python script generates a html page with a table of test set predictions. On running the script, it generates a file called prediction_grid.html.

# Attention_final.ipynb
This python notebook builds seq2seq models with attention.

It contains code to configure and run wandb sweeps for attention models.

It also configures the best attention model from sweep and computes test accuracy.

It viualises connectivity (i.e) when the model is decoding the i-th character in the output which is the input character that it is looking at ?

# Song_generation_using_GPT2.ipynb

This python notebook finetunes a GPT2 model to generate lyrics for English songs.
**dataset_1.txt** is the data with which we finetune the GPT2 model.

# predictions_vanilla
All the test data predictions made by the best vanilla model for can be found in this folder. 

# predictions_attention
All the test data predictions made by the best attention model can be found in this folder.
