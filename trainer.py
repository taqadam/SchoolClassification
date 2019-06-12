class Trainer():
    def __init__(self, datagenTrain, datagenTest, modelBundle, tensorboardCallbacks=False):
        self.datagenTrain = datagenTrain
        self.datagenTest = datagenTest
        self.modelBundle = modelBundle
        self.tensorboardCallbacks = tensorboardCallbacks

    def train(self, epochs, callbacks):
        if not self.tensorboardCallbacks:
            callbacks = []
        self.modelBundle.model.fit_generator(generator=self.datagenTrain,
                            validation_data=self.datagenTest,
                            use_multiprocessing=False,
                            workers=1,
                            callbacks=callbacks,
                            epochs = epochs)
