# Venture Funding with Deep Learning

You work as a risk management associate at Alphabet Soup, a venture capital firm. Alphabet Soup’s business team receives many funding applications from startups every day. This team has asked you to help them create a model that predicts whether applicants will be successful if funded by Alphabet Soup.

The business team has given you a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. With your knowledge of machine learning and neural networks, you decide to use the features in the provided dataset to create a binary classifier model that will predict whether an applicant will become a successful business. The CSV file contains a variety of information about these businesses, including whether or not they ultimately became successful.


Data Source: `Resources/applicantd_data.csv`

Data Pre-Processing:
* Drop columns that are not useful for modelling (EIN | NAME)
* Using `OneHotEncoder` for categorical labelling in each column
* Split Data into Train and Test
* Feature Scaling > Note: Vanishing gradient problem
* Create NN using `Sequential` model with `relu` activation in hidden layer and `sigmoid` in binary output (1 neuron)
* Compile() 
    ```
    loss='binary_crossentropy'
    optimizer='adam'
    metrics='accuracy'
    ```
* Fit()
    ```
    epochs=50
    batch_size=8
    ```
```
Epoch 1/50
3216/3216 [==============================] - 9s 3ms/step - loss: 0.5434 - accuracy: 0.7357
Epoch 2/50
3216/3216 [==============================] - 8s 2ms/step - loss: 0.5428 - accuracy: 0.7351
Epoch 3/50
3216/3216 [==============================] - 7s 2ms/step - loss: 0.5424 - accuracy: 0.7356
Epoch 4/50
3216/3216 [==============================] - 5s 2ms/step - loss: 0.5415 - accuracy: 0.7364
Epoch 5/50
3216/3216 [==============================] - 5s 2ms/step - loss: 0.5412 - accuracy: 0.7367
Epoch 6/50
3216/3216 [==============================] - 7s 2ms/step - loss: 0.5405 - accuracy: 0.7366
Epoch 7/50
3216/3216 [==============================] - 5s 2ms/step - loss: 0.5399 - accuracy: 0.7372
Epoch 8/50
3216/3216 [==============================] - 4s 1ms/step - loss: 0.5402 - accuracy: 0.7362
Epoch 9/50
3216/3216 [==============================] - 4s 1ms/step - loss: 0.5394 - accuracy: 0.7374
Epoch 10/50
3216/3216 [==============================] - 6s 2ms/step - loss: 0.5395 - accuracy: 0.7367
Epoch 11/50
3216/3216 [==============================] - 8s 2ms/step - loss: 0.5391 - accuracy: 0.7375
Epoch 12/50
3216/3216 [==============================] - 8s 2ms/step - loss: 0.5390 - accuracy: 0.7358
Epoch 13/50
3216/3216 [==============================] - 8s 3ms/step - loss: 0.5385 - accuracy: 0.7375
Epoch 14/50
3216/3216 [==============================] - 9s 3ms/step - loss: 0.5384 - accuracy: 0.7383
Epoch 15/50
3216/3216 [==============================] - 7s 2ms/step - loss: 0.5382 - accuracy: 0.7382
Epoch 16/50
3216/3216 [==============================] - 6s 2ms/step - loss: 0.5379 - accuracy: 0.7383
Epoch 17/50
3216/3216 [==============================] - 4s 1ms/step - loss: 0.5378 - accuracy: 0.7374
Epoch 18/50
3216/3216 [==============================] - 8s 2ms/step - loss: 0.5378 - accuracy: 0.7371
Epoch 19/50
3216/3216 [==============================] - 8s 2ms/step - loss: 0.5378 - accuracy: 0.7376
Epoch 20/50
3216/3216 [==============================] - 8s 2ms/step - loss: 0.5370 - accuracy: 0.7378
Epoch 21/50
3216/3216 [==============================] - 9s 3ms/step - loss: 0.5372 - accuracy: 0.7376
Epoch 22/50
3216/3216 [==============================] - 9s 3ms/step - loss: 0.5369 - accuracy: 0.7385
Epoch 23/50
3216/3216 [==============================] - 9s 3ms/step - loss: 0.5362 - accuracy: 0.7394
Epoch 24/50
3216/3216 [==============================] - 8s 3ms/step - loss: 0.5362 - accuracy: 0.7373
Epoch 25/50
3216/3216 [==============================] - 8s 2ms/step - loss: 0.5364 - accuracy: 0.7388
Epoch 26/50
3216/3216 [==============================] - 8s 2ms/step - loss: 0.5359 - accuracy: 0.7390
Epoch 27/50
3216/3216 [==============================] - 8s 3ms/step - loss: 0.5356 - accuracy: 0.7405
Epoch 28/50
3216/3216 [==============================] - 8s 2ms/step - loss: 0.5355 - accuracy: 0.7393
Epoch 29/50
3216/3216 [==============================] - 8s 3ms/step - loss: 0.5356 - accuracy: 0.7391
Epoch 30/50
3216/3216 [==============================] - 8s 3ms/step - loss: 0.5351 - accuracy: 0.7396
Epoch 31/50
3216/3216 [==============================] - 8s 2ms/step - loss: 0.5354 - accuracy: 0.7397
Epoch 32/50
3216/3216 [==============================] - 9s 3ms/step - loss: 0.5351 - accuracy: 0.7387
Epoch 33/50
3216/3216 [==============================] - 8s 2ms/step - loss: 0.5343 - accuracy: 0.7398
Epoch 34/50
3216/3216 [==============================] - 8s 2ms/step - loss: 0.5347 - accuracy: 0.7390
Epoch 35/50
3216/3216 [==============================] - 8s 3ms/step - loss: 0.5344 - accuracy: 0.7393
Epoch 36/50
3216/3216 [==============================] - 8s 3ms/step - loss: 0.5350 - accuracy: 0.7397
Epoch 37/50
3216/3216 [==============================] - 8s 3ms/step - loss: 0.5341 - accuracy: 0.7405
Epoch 38/50
3216/3216 [==============================] - 9s 3ms/step - loss: 0.5349 - accuracy: 0.7393
Epoch 39/50
3216/3216 [==============================] - 8s 3ms/step - loss: 0.5344 - accuracy: 0.7400
Epoch 40/50
3216/3216 [==============================] - 8s 3ms/step - loss: 0.5339 - accuracy: 0.7402
Epoch 41/50
3216/3216 [==============================] - 8s 2ms/step - loss: 0.5346 - accuracy: 0.7395
Epoch 42/50
3216/3216 [==============================] - 8s 2ms/step - loss: 0.5344 - accuracy: 0.7392
Epoch 43/50
3216/3216 [==============================] - 8s 3ms/step - loss: 0.5340 - accuracy: 0.7396
Epoch 44/50
3216/3216 [==============================] - 7s 2ms/step - loss: 0.5337 - accuracy: 0.7386
Epoch 45/50
3216/3216 [==============================] - 9s 3ms/step - loss: 0.5338 - accuracy: 0.7402
Epoch 46/50
3216/3216 [==============================] - 8s 3ms/step - loss: 0.5336 - accuracy: 0.7403
Epoch 47/50
3216/3216 [==============================] - 8s 3ms/step - loss: 0.5335 - accuracy: 0.7403
Epoch 48/50
3216/3216 [==============================] - 9s 3ms/step - loss: 0.5333 - accuracy: 0.7402
Epoch 49/50
3216/3216 [==============================] - 8s 2ms/step - loss: 0.5334 - accuracy: 0.7395
Epoch 50/50
3216/3216 [==============================] - 8s 3ms/step - loss: 0.5326 - accuracy: 0.7401
<keras.callbacks.History at 0x7f17ac5dc280>


# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test)# YOUR CODE HERE

>>> 268/268 [==============================] - 0s 407us/step - loss: 0.5623 - accuracy: 0.7303

# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

>>> Loss: 0.5622900128364563, Accuracy: 0.7302623987197876
```

* Save model >> `Resources/AlphabetSoup.h5`

Optimization Attempts:
* Review the notebook to view Alternative Model Results

All models are saved in `Resources/`


