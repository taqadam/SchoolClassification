import io
from PIL import Image
import keras
import tensorflow as tf
import numpy as np
import cv2

class customTensorboardInfo(keras.callbacks.Callback):
    def __init__(self, imageScale, tag, logdir, validationGenerator):
        super().__init__()
        self.imageScale = imageScale
        self.tag = tag
        self.logdir = logdir
        self.validationGenerator = validationGenerator
        if tf.gfile.Exists(self.logdir):
            tf.gfile.DeleteRecursively(self.logdir)

    def convertToProtobuff(self, tensor):

        height, width, channels = tensor.shape
        image = Image.fromarray(tensor[:,:,::-1])
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                             width=width,
                             colorspace=channels,
                             encoded_image_string=image_string)

    def confusionMatrix(self, epoch):
        cMatrix = np.zeros((2,2))

        for i in range(len(self.validationGenerator)):
            validationData = self.validationGenerator.__getitem__(i)
            predictions = self.model.predict(validationData[0])
            predictions = np.argmax(predictions,axis=1)
            truth = np.argmax(validationData[1],axis=1)

            for j in range(validationData[1].shape[0]):
                cMatrix[truth[j],predictions[j]] += 1

        writer = tf.summary.FileWriter(self.logdir)

        header_row = 'T \ P | No school | School'
        headerColumn = tf.constant(["**No school**", "**School**"])
        noschoolColumn = tf.as_string(tf.convert_to_tensor(cMatrix[:,0]))
        schoolColumn = tf.as_string(tf.convert_to_tensor(cMatrix[:,1]))
        table_rows = tf.strings.join([headerColumn," | ",noschoolColumn," | ",schoolColumn])
        table_body = tf.strings.reduce_join(inputs=table_rows, separator="\n")
        table = tf.strings.join([header_row, "---|---|---",table_body], separator='\n')

        summary_op = tf.summary.text('Confusion Matrix for Epoch {}'.format(epoch), table)
        with tf.Session() as sess:
            summary = sess.run(summary_op)
            writer.add_summary(summary)
        writer.close()

    def on_epoch_begin(self, epoch, logs={}):
        self.saveModel(epoch)
        self.confusionMatrix(epoch)
        self.imageConstruction(epoch)

    def saveModel(self, epoch):
        if epoch % 10 == 0:
            self.model.save_weights('./models/modelBundle_{}_epoch_{}'.format(self.imageScale, epoch))

    def imageConstruction(self, epoch, logs={}):

        validationDatum = self.validationGenerator.__getitem__(0)

        writer = tf.summary.FileWriter(self.logdir)
        predictions = self.model.predict(validationDatum[0])
        for i in range(8):
            imgCopy = np.copy(validationDatum[0][i])
            imgCopy = (np.array(self.validationGenerator.std) * (imgCopy)+np.array(self.validationGenerator.mean)).astype('uint8')
            # possible future contrast augmentations may put some values beyond the range (0,255)
            imgCopy = np.clip(imgCopy, 0, 255)

            label = np.argmax(validationDatum[1][i])
            predLabel = np.argmax(predictions[i])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(imgCopy,"T:"+str(label)+" P:"+str(predLabel),(8,32), font, 0.5,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(imgCopy,"T:"+str(label)+" P:"+str(predLabel),(10,32), font, 0.5,(0,0,0),2,cv2.LINE_AA)
            image = self.convertToProtobuff(imgCopy)
            summary = tf.Summary(value=[tf.Summary.Value(tag='Visualizing Epoch {}'.format(epoch), image=image)])
            writer.add_summary(summary, epoch)
        writer.close()
