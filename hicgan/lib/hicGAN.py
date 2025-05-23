import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv1D, BatchNormalization, LeakyReLU, Conv2DTranspose, Dropout, ReLU, Flatten, Dense
import numpy as np
import os
import matplotlib.pyplot as plt 
from tqdm import tqdm
from . import utils

#implementation of adapted pix2pix cGAN
#modified from tensorflow tutorial https://www.tensorflow.org/tutorials/generative/pix
#also see: https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/


class HiCGAN():
    def __init__(self, log_dir: str, 
                    number_factors: int,
                    loss_weight_pixel: float = 100, #factor for L1/L2 loss, like Isola et al. 2017
                    loss_weight_adversarial: float = 1.0, #factor for adversarial loss in generator
                    loss_weight_discriminator: float = 0.5, #factor for disc loss, like Isola et al. 2017
                    loss_weight_tv: float = 1e-10, #factor for TV loss
                    loss_type_pixel: str = "L1", #type of per-pixel error (L1, L2)
                    input_size: int = 256,
                    plot_frequency: int = 20,
                    plot_type: str = "png",
                    learning_rate_generator: float = 2e-5,
                    learning_rate_discriminator: float = 1e-6,
                    adam_beta_1: float = 0.5,
                    pretrained_model_path: str = "",
                    scope=None): 
        super().__init__()

        self.OUTPUT_CHANNELS = 1
        self.INPUT_CHANNELS = 1
        self.input_size = 256
        if input_size in [64,128,256, 512]:
            self.input_size = input_size
        self.number_factors = number_factors
        self.loss_weight_pixel = loss_weight_pixel
        self.loss_weight_discriminator = loss_weight_discriminator
        self.loss_weight_adversarial = loss_weight_adversarial
        self.loss_weight_tv = loss_weight_tv
        self.loss_type_pixel = loss_type_pixel
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_generator, beta_1=adam_beta_1, name="Adam_Generator")
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_discriminator, beta_1=adam_beta_1, name="Adam_Discriminator")

        self.generator_embedding = self.cnn_embedding()
        self.discriminator_embedding = self.cnn_embedding()         
        self.generator = self.Generator()
        self.discriminator = self.Discriminator()

        self.log_dir=log_dir
        
        self.checkpoint_dir = os.path.join(self.log_dir, 'training_checkpoints')
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                    discriminator_optimizer=self.discriminator_optimizer,
                                    generator=self.generator,
                                    discriminator=self.discriminator)
        self.plot_type = plot_type
        if self.plot_type not in ["png", "pdf", "svg"]:
            self.plot_type = "png"
            print("plot type {:s} unsupported, changed to png".format(plot_type))
        self.progress_plot_name = os.path.join(self.log_dir, "lossOverEpochs.{:s}".format(self.plot_type))
        self.progress_plot_frequency = plot_frequency
        self.example_plot_frequency = 5

        #losses per epochs
        self.__gen_train_loss_epochs = []
        self.__disc_train_loss_true_epochs = []
        self.__disc_train_loss_fake_epochs = []
        self.__gen_val_loss_epochs = []
        self.__disc_val_loss_epochs = []
        #losses per batch
        self.__gen_train_loss_batches = []
        self.__disc_train_loss_true_batches = []
        self.__disc_train_loss_fake_batches = []
        self.__gen_val_loss_batches = []
        self.__disc_val_loss_batches = []

        self.__epoch_counter = 0
        self.__batch_counter = 0

        self.scope = scope

    def cnn_embedding(self, nr_filters_list=[1024,512,512,256,256,128,128,64], kernel_width_list=[4,4,4,4,4,4,4,4], apply_dropout: bool = False):  
        inputs = tf.keras.layers.Input(shape=(3*self.input_size, self.number_factors))
        #add 1D convolutions
        x = inputs
        for i, (nr_filters, kernelWidth) in enumerate(zip(nr_filters_list, kernel_width_list)):
            convParamDict = dict()
            convParamDict["name"] = "conv1D_" + str(i + 1)
            convParamDict["filters"] = nr_filters
            convParamDict["kernel_size"] = kernelWidth
            convParamDict["data_format"]="channels_last"
            convParamDict["kernel_regularizer"]=tf.keras.regularizers.l2(0.01)
            if kernelWidth > 1:
                convParamDict["padding"] = "same"
            x = Conv1D(**convParamDict)(x)
            x = BatchNormalization()(x)
            if apply_dropout:
                x = Dropout(0.5)(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        #make the shape of a square matrix
        x = Conv1D(filters=self.input_size, 
                    strides=3, 
                    kernel_size=4, 
                    data_format="channels_last", 
                    activation="sigmoid", 
                    padding="same", name="conv1D_final")(x)
        #ensure the matrix is symmetric, i.e. x = transpose(x)
        x_T = tf.keras.layers.Permute((2,1))(x) #this is the matrix transpose
        x = tf.keras.layers.Add()([x, x_T])
        x = tf.keras.layers.Lambda(lambda z: 0.5*z)(x) #add transpose and divide by 2
        #reshape the matrix into a 2D grayscale image
        x = tf.keras.layers.Reshape((self.input_size,self.input_size,self.INPUT_CHANNELS))(x)
        model = tf.keras.Model(inputs=inputs, outputs=x, name="CNN-embedding")
        #model.build(input_shape=(3*self.INPUT_SIZE, self.NR_FACTORS))
        #model.summary()
        return model


    @staticmethod
    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm:
            result.add(BatchNormalization())
        result.add(LeakyReLU(alpha=0.2))
        return result

    @staticmethod
    def upsample(filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))
        result.add(BatchNormalization())
        if apply_dropout:
            result.add(Dropout(0.5))
        result.add(ReLU())
        return result


    def Generator(self):
        inputs = tf.keras.layers.Input(shape=[3*self.input_size,self.number_factors], name="factorData")

        twoD_conversion = self.generator_embedding
        #the downsampling part of the network, defined for 256x256 images
        down_stack = [
            HiCGAN.downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
            HiCGAN.downsample(128, 4), # (bs, 64, 64, 128)
            HiCGAN.downsample(256, 4), # (bs, 32, 32, 256)
            HiCGAN.downsample(512, 4), # (bs, 16, 16, 512)
            HiCGAN.downsample(512, 4), # (bs, 8, 8, 512)
            HiCGAN.downsample(512, 4), # (bs, 4, 4, 512)
            HiCGAN.downsample(512, 4), # (bs, 2, 2, 512)
            HiCGAN.downsample(512, 4, apply_batchnorm=False), # (bs, 1, 1, 512)
        ]
        #if the input images are smaller, leave out some layers accordingly
        if self.input_size < 256:
            down_stack = down_stack[:-2] + down_stack[-1:]
        if self.input_size < 128:
            down_stack = down_stack[:-2] + down_stack[-1:]

        #the upsampling portion of the generator, designed for 256x256 images
        up_stack = [
            HiCGAN.upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
            HiCGAN.upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
            HiCGAN.upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
            HiCGAN.upsample(512, 4), # (bs, 16, 16, 1024)
            HiCGAN.upsample(256, 4), # (bs, 32, 32, 512)
            HiCGAN.upsample(128, 4), # (bs, 64, 64, 256)
            HiCGAN.upsample(64, 4), # (bs, 128, 128, 128)
        ]
        #for smaller images, take layers away, otherwise downsampling won't work
        if self.input_size < 256:
            up_stack = up_stack[:2] + up_stack[3:]
        if self.input_size < 128:
            up_stack = up_stack[:2] + up_stack[3:]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.OUTPUT_CHANNELS, 4,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=initializer) # (bs, 256, 256, 3)

        x = inputs
        x = twoD_conversion(x)

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)
        #enforce symmetry
        x_T = tf.keras.layers.Permute((2,1,3))(x)
        x = tf.keras.layers.Add()([x, x_T])
        x = tf.keras.layers.Lambda(lambda z: 0.5*z)(x)
        x = tf.keras.layers.Activation("sigmoid")(x)

        return tf.keras.Model(inputs=inputs, outputs=x, name="Generator")


    def generator_loss(self, disc_generated_output, gen_output, target):
        advers_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_generated_output), logits=disc_generated_output) )
        # mean squared error or mean absolute error
        if self.loss_type_pixel == "L1":
            pixel_loss = tf.reduce_mean(tf.abs(target - gen_output))
        else: 
            pixel_loss = tf.reduce_mean(tf.square(target - gen_output))
        tv_loss = tf.reduce_mean(tf.image.total_variation(gen_output))
        total_gen_loss = self.loss_weight_pixel * pixel_loss + self.loss_weight_adversarial * advers_loss + self.loss_weight_tv * tv_loss
        return total_gen_loss, advers_loss, pixel_loss


    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[3*self.input_size, self.number_factors], name='input_image')
        tar = tf.keras.layers.Input(shape=[self.input_size, self.input_size, self.OUTPUT_CHANNELS], name='target_image')
        embedding = self.discriminator_embedding
        #Patch-GAN (Isola et al.)
        d = embedding(inp)
        d = tf.keras.layers.Concatenate()([d, tar])
        if self.input_size > 64:
            #downsample and symmetrize 1 
            d = HiCGAN.downsample(64, 4, False)(d) # (bs, inp.size/2, inp.size/2, 64)
            d_T = tf.keras.layers.Permute((2,1,3))(d)
            d = tf.keras.layers.Add()([d, d_T])
            d = tf.keras.layers.Lambda(lambda z: 0.5*z)(d)
            #downsample and symmetrize 2
            d = HiCGAN.downsample(128, 4)(d)# (bs, inp.size/4, inp.size/4, 128)
            d_T = tf.keras.layers.Permute((2,1,3))(d)
            d = tf.keras.layers.Add()([d, d_T])
            d = tf.keras.layers.Lambda(lambda z: 0.5*z)(d)
        else:    
            #downsample and symmetrize 3
            d = HiCGAN.downsample(256, 4)(d)
            d_T = tf.keras.layers.Permute((2,1,3))(d)
            d = tf.keras.layers.Add()([d, d_T])
            d = tf.keras.layers.Lambda(lambda z: 0.5*z)(d)
        #downsample and symmetrize 4
        d = HiCGAN.downsample(256, 4)(d) # (bs, inp.size/8, inp.size/8, 256)
        d_T = tf.keras.layers.Permute((2,1,3))(d)
        d = tf.keras.layers.Add()([d, d_T])
        d = tf.keras.layers.Lambda(lambda z: 0.5*z)(d)
        d = Conv2D(512, 4, strides=1, padding="same", kernel_initializer=initializer)(d) #(bs, inp.size/8, inp.size/8, 512)
        d_T = tf.keras.layers.Permute((2,1,3))(d)
        d = tf.keras.layers.Add()([d, d_T])
        d = tf.keras.layers.Lambda(lambda z: 0.5*z)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(1, 4, strides=1, padding="same",
                        kernel_initializer=initializer)(d) #(bs, inp.size/8, inp.size/8, 1)
        d_T = tf.keras.layers.Permute((2,1,3))(d)
        d = tf.keras.layers.Add()([d, d_T])
        d = tf.keras.layers.Lambda(lambda z: 0.5*z)(d)
        #d = tf.keras.layers.Activation("sigmoid")(d) #sigmoid will be done in the loss function itself
        return tf.keras.Model(inputs=[inp, tar], outputs=d, name="Discriminator")

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_real_output), logits=disc_real_output) )
        generated_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(disc_generated_output), logits=disc_generated_output) )
        total_disc_loss = self.loss_weight_discriminator * (real_loss + generated_loss)
        return total_disc_loss, real_loss, generated_loss

    @tf.function
    def distributed_train_step(self, input_data):
        input_image, target = input_data[0]["factorData"], input_data[1]["out_matrixData"]
        per_replica_losses = self.scope.run(self.train_step, args=(input_image, target, ))
        return self.scope.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    
    @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, _, _ = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss, disc_real_loss, disc_gen_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    self.discriminator.trainable_variables))
        return gen_total_loss, disc_loss, disc_real_loss, disc_gen_loss

    @tf.function
    def validationStep(self, input_image, target):
        gen_output = self.generator(input_image, training=True)

        disc_real_output = self.discriminator([input_image, target], training=True)
        disc_generated_output = self.discriminator([input_image, gen_output], training=True)

        gen_total_loss, _, _ = self.generator_loss(disc_generated_output, gen_output, target)
        disc_loss, _, _ = self.discriminator_loss(disc_real_output, disc_generated_output)
        return gen_total_loss, disc_loss

    def generate_images(self, model, test_input, tar, epoch: int):
        prediction = model(test_input, training=True)
        pred_mse = tf.reduce_mean(tf.square( tar["out_matrixData"][0], prediction[0] ))
        figname = "testpred_epoch_{:05d}.png".format(epoch)
        figname = os.path.join(self.log_dir, figname)
        display_list = [test_input["factorData"][0], tar["out_matrixData"][0], prediction[0]]
        titleList = ['Input Image', 'Ground Truth', 'Predicted Image (MSE: {:.5f})'.format(pred_mse)]

        fig1, axs1 = plt.subplots(1,len(display_list), figsize=(15,15))
        for i in range(len(display_list)):
            axs1[i].imshow(display_list[i] * 0.5 + 0.5)
            axs1[i].set_title(titleList[i])
        fig1.savefig(figname)
        plt.close(fig1)
        del fig1, axs1

    def fit(self, train_ds, epochs, test_ds, steps_per_epoch: int):
        distributed_dataset = self.scope.experimental_distribute_dataset(train_ds)

        for epoch in range(epochs):
            #generate sample output
            if epoch % self.example_plot_frequency == 0:
                for example_input, example_target in test_ds.take(1):
                    self.generate_images(self.generator, example_input, example_target, epoch)
            
            
            train_pbar = tqdm(total=steps_per_epoch)
            train_pbar.set_description("Epoch {:05d}".format(epoch+1))
            train_samples_in_epoch = 0

            for distributed_input in distributed_dataset:
               
                train_samples_in_epoch += 1
                gen_loss, _, disc_loss_real, disc_loss_fake = self.distributed_train_step(distributed_input)

                self.__disc_train_loss_true_batches.append(disc_loss_real)
                self.__disc_train_loss_fake_batches.append(disc_loss_fake)
                self.__gen_train_loss_batches.append(gen_loss)
                train_bar_postfixDict= {"g": "{:.4f}".format(self.__gen_train_loss_batches[-1]),
                                         "dt": "{:.3f}".format(self.__disc_train_loss_true_batches[-1]),
                                         "df": "{:.3f}".format(self.__disc_train_loss_fake_batches[-1])}
                if len(self.__gen_val_loss_epochs) > 0:
                    train_bar_postfixDict["v"] = "{:.3f}".format(self.__gen_val_loss_epochs[-1])
                train_pbar.set_postfix( train_bar_postfixDict )
                train_pbar.update(1)
                self.__batch_counter += 1
            train_pbar.close()
            self.__gen_train_loss_epochs.append(np.mean(self.__gen_train_loss_batches[-train_samples_in_epoch:]))
            self.__disc_train_loss_true_epochs.append(np.mean(self.__disc_train_loss_true_batches[-train_samples_in_epoch:]))
            self.__disc_train_loss_fake_epochs.append(np.mean(self.__disc_train_loss_fake_batches[-train_samples_in_epoch:]))

            # Validation
            validation_samples_in_epoch = 0
            for input_image, target in test_ds:
                gen_loss_val, disc_loss_val = self.validationStep(input_image["factorData"], target["out_matrixData"])
                self.__gen_val_loss_batches.append(gen_loss_val)
                self.__disc_val_loss_batches.append(disc_loss_val)
                validation_samples_in_epoch += 1
            self.__disc_val_loss_epochs.append(np.mean(self.__disc_val_loss_batches[-validation_samples_in_epoch:]))
            self.__gen_val_loss_epochs.append(np.mean(self.__gen_val_loss_batches[-validation_samples_in_epoch:]))

            #count the epoch
            self.__epoch_counter += 1

            # saving the model every 20 epochs
            if (epoch + 1) % self.progress_plot_frequency == 0:
                #plot loss
                utils.plotLoss(pGeneratorLossValueLists=[self.__gen_train_loss_epochs, self.__gen_val_loss_epochs],
                              pDiscLossValueLists=[ [self.loss_weight_discriminator*sum(x) for x in zip(self.__disc_train_loss_fake_epochs, self.__disc_train_loss_true_epochs)],
                                                    self.__disc_train_loss_true_epochs, 
                                                    self.__disc_train_loss_fake_epochs, 
                                                    self.__disc_val_loss_epochs],
                              pGeneratorLossNameList=["training", "validation"],
                              pDiscLossNameList=["train total", "train real", "train gen.", "valid. total"],
                              pFilename=self.progress_plot_name,
                              useLogscaleList=[True, False])
                np.savez(os.path.join(self.log_dir, "lossValues_{:05d}.npz".format(epoch)), 
                                    genLossTrain=self.__gen_train_loss_batches, 
                                    genLossVal=self.__gen_val_loss_batches, 
                                    discLossTrain_True=self.__disc_train_loss_true_batches,
                                    discLossTrain_Fake=self.__disc_train_loss_fake_batches, 
                                    discLossVal=self.__disc_val_loss_batches)
                self.generator.save(filepath=os.path.join(self.log_dir, "generator_{:05d}.keras".format(epoch)), save_format="keras")
                self.discriminator.save(filepath=os.path.join(self.log_dir, "discriminator_{:05d}.keras".format(epoch)), save_format="keras")


        self.checkpoint.save(file_prefix = self.checkpoint_prefix)
        utils.plotLoss(pGeneratorLossValueLists=[self.__gen_train_loss_epochs, self.__gen_val_loss_epochs],
                              pDiscLossValueLists=[ [self.loss_weight_discriminator*sum(x) for x in zip(self.__disc_train_loss_fake_epochs, self.__disc_train_loss_true_epochs)],
                                                    self.__disc_train_loss_true_epochs, 
                                                    self.__disc_train_loss_fake_epochs, 
                                                    self.__disc_val_loss_epochs],
                              pGeneratorLossNameList=["training", "validation"],
                              pDiscLossNameList=["train total", "train real", "train gen.", "valid. total"],
                              pFilename=self.progress_plot_name,
                              useLogscaleList=[True, False])
        np.savez(os.path.join(self.log_dir, "lossValues_{:05d}.npz".format(epoch)), 
                                    genLossTrain=self.__gen_train_loss_batches, 
                                    genLossVal=self.__gen_val_loss_batches, 
                                    discLossTrain_True=self.__disc_train_loss_true_batches,
                                    discLossTrain_Fake=self.__disc_train_loss_fake_batches, 
                                    discLossVal=self.__disc_val_loss_batches)
        self.generator.save(filepath=os.path.join(self.log_dir, "generator_{:05d}.keras".format(epoch)), save_format="keras")
        self.discriminator.save(filepath=os.path.join(self.log_dir, "discriminator_{:05d}.keras".format(epoch)), save_format="keras")

    def plotModels(self, pOutputPath: str, pFigureFileFormat: str):
        generatorPlotName = "generatorModel.{:s}".format(pFigureFileFormat)
        generatorPlotName = os.path.join(pOutputPath, generatorPlotName)
        discriminatorPlotName = "discriminatorModel.{:s}".format(pFigureFileFormat)
        discriminatorPlotName = os.path.join(pOutputPath, discriminatorPlotName)
        generatorEmbeddingPlotName = "generatorEmbeddingModel.{:s}".format(pFigureFileFormat)
        generatorEmbeddingPlotName = os.path.join(pOutputPath, generatorEmbeddingPlotName)
        discriminatorEmbeddingPlotName = "discriminatorEmbeddingModel.{:s}".format(pFigureFileFormat)
        discriminatorEmbeddingPlotName = os.path.join(pOutputPath, discriminatorEmbeddingPlotName)
        tf.keras.utils.plot_model(self.generator, show_shapes=True, to_file=generatorPlotName)
        tf.keras.utils.plot_model(self.discriminator, show_shapes=True, to_file=discriminatorPlotName)
        tf.keras.utils.plot_model(self.generator_embedding, show_shapes=True, to_file=generatorEmbeddingPlotName)
        tf.keras.utils.plot_model(self.discriminator_embedding, show_shapes=True, to_file=discriminatorEmbeddingPlotName)

    def predict(self, test_ds, steps_per_record):
        predictedArray = []
        for batch in tqdm(test_ds, desc="Predicting", total=steps_per_record):
            predBatch = self.predictionStep(input_batch=batch).numpy()
            for i in range(predBatch.shape[0]):
                predictedArray.append(predBatch[i][:,:,0])        
        predictedArray = np.array(predictedArray)
        return predictedArray
    
    @tf.function
    def predictionStep(self, input_batch, training=True):
        return self.generator(input_batch, training=training)
  
    
    def loadGenerator(self, trainedModelPath: str):
        '''
            load a trained generator model for prediction
        '''
        try:
            trainedModel = tf.keras.models.load_model(filepath=trainedModelPath, 
                                                  custom_objects={"CustomReshapeLayer": CustomReshapeLayer(self.input_size)}, safe_mode=False)
            self.generator = trainedModel
        except Exception as e:
            msg = str(e)
            msg += "\nError: failed to load trained model"
            raise ValueError(msg)

class CustomReshapeLayer(tf.keras.layers.Layer):
    '''
    reshape a 1D tensor such that it represents 
    the upper triangular part of a square 2D matrix with shape (matsize, matsize)
    #example: 
     [1,2,3,4,5,6] => [[1,2,3],
                       [0,4,5],
                       [0,0,6]]
    '''
    def __init__(self, matsize, **kwargs):
        super(CustomReshapeLayer, self).__init__(**kwargs)
        self.matsize = matsize
        self.triu_indices = [ [x,y] for x,y in zip(np.triu_indices(self.matsize)[0], np.triu_indices(self.matsize)[1]) ]

    def call(self, inputs):      
        return tf.map_fn(self.pickItems, inputs, parallel_iterations=20, swap_memory=True)
        
    def pickItems(self, inputVec):
        sparseTriuTens = tf.SparseTensor(self.triu_indices, 
                                        values=inputVec, 
                                        dense_shape=[self.matsize, self.matsize] )
        return tf.sparse.to_dense(sparseTriuTens)

    def get_config(self):
        return {"matsize": self.matsize}