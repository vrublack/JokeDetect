from train_audio import make_model

model, X, Y = make_model()

model.load_weights('weights_audio')

predict = model.predict(X[60:70])

print('Predictions: \n' + str(predict))

print('Actual Y: \n' + str(Y[60:70]))