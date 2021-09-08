import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Concatenate, Dropout, Input
from tensorflow.keras.layers import ZeroPadding2D,Conv2DTranspose,LeakyReLU
from tensorflow.keras.layers import Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
from tensorflow.nn import max_pool_with_argmax
from tensorflow_addons.layers import MaxUnpooling2D

def get_colorization_model(model_name):
    
    if model_name=="AE":
        model = AE(RESNET_STRUCTURE=True).get_model()
        model.compile(loss="mse", optimizer="adam", metrics=["mse"])
        
    elif model_name=="SegNet":
        model = SegNet(RESNET_STRUCTURE=True).get_model()
        model.compile(loss="mse", optimizer="adam", metrics=["mse"])
        
    elif model_name=="Unet":
        model = Unet(input_ch=1, output_ch=2).get_model()
        model.compile(loss="mse", optimizer="adam", metrics=["mse"])
        
    return model


class EarlyStopping:
    """Earlystopping for Colorization Network"""

    def __init__(self, patience=20):
        self.patience = patience    
        self.counter = 0            
        self.min_loss = np.Inf      
        self.early_stop = False     

    def __call__(self, loss):

        if loss > self.min_loss:  
            self.counter += 1   
            if self.counter >= self.patience:  
                self.early_stop = True
        else:  # when achive best score
            self.min_loss = loss  
            self.counter = 0  


class AE(object):

    def __init__(self, input_ch=1, output_ch=2, filters=64*1, INPUT_IMAGE_SIZE=256, RESNET_STRUCTURE=True):
        self.INPUT_CH = input_ch 
        self.OUTPUT_CH = output_ch
        self.FILTERS = filters
        self.INPUT_IMAGE_SIZE = INPUT_IMAGE_SIZE
        self.RESNET_STRUCTURE = RESNET_STRUCTURE

        self.AE = self.build_model()
    
    def build_model(self):

        def conv2d(layer_input, filters, f_size=3, stride=2, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=stride, padding='same')(layer_input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        def resnet_conv2d(layer_input, filters, f_size=3, stride=1, bn=True):
            """Layers used between downsampling and upsampling """
            d = Conv2D(filters, kernel_size=f_size, strides=stride, padding='same')(layer_input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)

            d = Conv2D(filters, kernel_size=f_size, strides=stride, padding='same')(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            
            d = d + layer_input
            return d 

        def deconv2d(layer_input, filters, f_size=4, stride=2, bn=True):
            """Layers used during upsampling"""
            u = Conv2DTranspose(filters, kernel_size=f_size, strides=stride, padding='same')(layer_input)
            if bn:
                u = BatchNormalization(momentum=0.8)(u)
            u = LeakyReLU(alpha=0.2)(u)
            return u
        
        
        # Image input
        inputs = Input(shape=(self.INPUT_IMAGE_SIZE,self.INPUT_IMAGE_SIZE,self.INPUT_CH))
        x = Conv2D(32, kernel_size=1, strides=1, padding='same')(inputs)
        x = LeakyReLU(alpha=0.2)(x)
        
        """ Encoder """
        # Downsampling
        x = conv2d(x, self.FILTERS*1, f_size=4, stride=2, bn=False) # C64  1/4
        x = conv2d(x, self.FILTERS*2, f_size=4, stride=2, bn=True) # C128  1/8
        x = conv2d(x, self.FILTERS*4, f_size=4, stride=2, bn=True) # C256  1/16
        x = conv2d(x, self.FILTERS*8, f_size=4, stride=2, bn=True) # C512  1/32

        if self.RESNET_STRUCTURE:
            # Resnet block bottom layers
            x = resnet_conv2d(x, self.FILTERS*8) # C512
            x = resnet_conv2d(x, self.FILTERS*8) # C512
            x = resnet_conv2d(x, self.FILTERS*8) # C512
            x = resnet_conv2d(x, self.FILTERS*8) # C512
            x = resnet_conv2d(x, self.FILTERS*8) # C512
            
        """ Decoder """
        # Upsampling
        x = deconv2d(x, self.FILTERS*4, f_size=4, stride=2, bn=True)
        x = deconv2d(x, self.FILTERS*2, f_size=4, stride=2, bn=True) 
        x = deconv2d(x, self.FILTERS*1, f_size=4, stride=2, bn=True)
        x = deconv2d(x, 32,             f_size=4, stride=2, bn=False) 
        
        output_img = Conv2D(self.OUTPUT_CH, kernel_size=4, strides=1, padding="same", activation='sigmoid')(x)
        
        return Model(inputs, output_img)

    def get_model(self):
        return self.AE


class SegNet(object):

    def __init__(self, input_ch=1, output_ch=2, filters=64*1, INPUT_IMAGE_SIZE=256, RESNET_STRUCTURE=True):
        self.INPUT_CH = input_ch 
        self.OUTPUT_CH = output_ch
        self.FILTERS = filters
        self.INPUT_IMAGE_SIZE = INPUT_IMAGE_SIZE

        self.RESNET_STRUCTURE = RESNET_STRUCTURE
        self.SEGNET = self.build_model()
    
    def build_model(self):

        def conv_block(x, filters):
            
            x = Conv2D(filters, kernel_size=3, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            
            # x = Conv2D(filters, kernel_size=3, padding="same")(x)
            # x = BatchNormalization()(x)
            # x = Activation("relu")(x)
            
            return x
        
        def resnet_conv2d(layer_input, filters, f_size=3, stride=1, bn=True):
            """Layers used between downsampling and upsampling """
            
            d = Conv2D(filters, kernel_size=f_size, strides=stride, padding='same')(layer_input)
            d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            
            d = Conv2D(filters, kernel_size=f_size, strides=stride, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            
            d = d + layer_input
            
            return d 
                
        # Image input
        inputs = Input(shape=(self.INPUT_IMAGE_SIZE,self.INPUT_IMAGE_SIZE,self.INPUT_CH))
        
        """ Encoder """
        
        # Stage 1
        x = conv_block(inputs, self.FILTERS*1)
        x = conv_block(x, self.FILTERS*1)
        x,idx_1 = max_pool_with_argmax(x, ksize=2, strides=2, padding="VALID")
        
        # Stage 2
        x = conv_block(x, self.FILTERS*2)
        x = conv_block(x, self.FILTERS*2)
        x,idx_2 = max_pool_with_argmax(x, ksize=2, strides=2, padding="VALID")
        
        # Stage 3
        x = conv_block(x, self.FILTERS*4)
        x = conv_block(x, self.FILTERS*4)
        x = conv_block(x, self.FILTERS*4)
        x,idx_3 = max_pool_with_argmax(x, ksize=2, strides=2, padding="VALID")
        
        # Stage 4
        x = conv_block(x, self.FILTERS*8)
        x = conv_block(x, self.FILTERS*8)
        x = conv_block(x, self.FILTERS*8)
        x,idx_4 = max_pool_with_argmax(x, ksize=2, strides=2, padding="VALID")
        
        # Stage 5
        x = conv_block(x, self.FILTERS*8)
        x = conv_block(x, self.FILTERS*8)
        x = conv_block(x, self.FILTERS*8)
        x,idx_5 = max_pool_with_argmax(x, ksize=2, strides=2, padding="VALID")
        
        if self.RESNET_STRUCTURE:
            # Resnet block bottom layers
            x = resnet_conv2d(x, self.FILTERS*8) # C512
            x = resnet_conv2d(x, self.FILTERS*8) # C512
            x = resnet_conv2d(x, self.FILTERS*8) # C512
            x = resnet_conv2d(x, self.FILTERS*8) # C512
            x = resnet_conv2d(x, self.FILTERS*8) # C512
        
        """ Decoder """
        # Stage 5u
        x = MaxUnpooling2D(pool_size=2, strides=2, padding="SAME")(x,idx_5)
        x = conv_block(x, self.FILTERS*8)
        x = conv_block(x, self.FILTERS*8)
        x = conv_block(x, self.FILTERS*8)
        
        # Stage 4u
        x = MaxUnpooling2D(pool_size=2, strides=2, padding="SAME")(x,idx_4)
        x = conv_block(x, self.FILTERS*8)
        x = conv_block(x, self.FILTERS*8)
        x = conv_block(x, self.FILTERS*4)
        
        # Stage 3u
        x = MaxUnpooling2D(pool_size=2, strides=2, padding="SAME")(x,idx_3)
        x = conv_block(x, self.FILTERS*4)
        x = conv_block(x, self.FILTERS*4)
        x = conv_block(x, self.FILTERS*2)
        
        # Stage 2u
        x = MaxUnpooling2D(pool_size=2, strides=2, padding="SAME")(x,idx_2)
        x = conv_block(x, self.FILTERS*2)
        x = conv_block(x, self.FILTERS*1)
        
        # Stage 1u
        x = MaxUnpooling2D(pool_size=2, strides=2, padding="SAME")(x,idx_1)
        x = conv_block(x, self.FILTERS*1)
        x = conv_block(x, self.FILTERS*1)
        
        outputs = Conv2D(self.OUTPUT_CH, kernel_size=1, strides=1, padding="same", activation='sigmoid')(x)
        
        return Model(inputs, outputs)

    def get_model(self):
        return self.SEGNET


class Unet(object):
    """
        Pix2pix-like U-Net
    """
    def __init__(self, input_ch=1, output_ch=2, filters=64*1, INPUT_IMAGE_SIZE=256):
        self.INPUT_CH = input_ch 
        self.OUTPUT_CH = output_ch
        self.FILTERS = filters
        self.INPUT_IMAGE_SIZE = INPUT_IMAGE_SIZE
        self.CONV_FILTER_SIZE = 4
        self.CONV_STRIDE = 2
        self.CONV_PADDING = (1, 1)
        self.DECONV_FILTER_SIZE = 2
        self.DECONV_STRIDE = 2

        self.UNET = self.build_model()
    
    def build_model(self):

        # (256 x 256 x input_channel_count)
        inputs = Input(shape=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, self.INPUT_CH))

        """
            Encoder
        """
        # (128 x 128 x N)
        enc1 = ZeroPadding2D(self.CONV_PADDING)(inputs)
        enc1 = Conv2D(self.FILTERS, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(enc1)

        # (64 x 64 x 2N)
        enc2 = self._add_encoding_layer(self.FILTERS*2, enc1)

        # (32 x 32 x 4N)
        enc3 = self._add_encoding_layer(self.FILTERS*4, enc2)

        # (16 x 16 x 8N)
        enc4 = self._add_encoding_layer(self.FILTERS*8, enc3)

        # (8 x 8 x 8N)
        enc5 = self._add_encoding_layer(self.FILTERS*8, enc4)

        # (4 x 4 x 8N)
        enc6 = self._add_encoding_layer(self.FILTERS*8, enc5)

        # (2 x 2 x 8N)
        enc7 = self._add_encoding_layer(self.FILTERS*8, enc6)

        # (1 x 1 x 8N)
        enc8 = self._add_encoding_layer(self.FILTERS*8, enc7)

        """
            Decoder
        """
        # (2 x 2 x 8N)
        dec1 = self._add_decoding_layer(self.FILTERS*8, True, enc8)
        dec1 = Concatenate(axis=-1)([dec1, enc7])

        # (4 x 4 x 8N)
        dec2 = self._add_decoding_layer(self.FILTERS*8, True, dec1)
        dec2 = Concatenate(axis=-1)([dec2, enc6])

        # (8 x 8 x 8N)
        dec3 = self._add_decoding_layer(self.FILTERS*8, True, dec2)
        dec3 = Concatenate(axis=-1)([dec3, enc5])

        # (16 x 16 x 8N)
        dec4 = self._add_decoding_layer(self.FILTERS*8, False, dec3)
        dec4 = Concatenate(axis=-1)([dec4, enc4])

        # (32 x 32 x 4N)
        dec5 = self._add_decoding_layer(self.FILTERS*4, False, dec4)
        dec5 = Concatenate(axis=-1)([dec5, enc3])

        # (64 x 64 x 2N)
        dec6 = self._add_decoding_layer(self.FILTERS*2, False, dec5)
        dec6 = Concatenate(axis=-1)([dec6, enc2])

        # (128 x 128 x N)
        dec7 = self._add_decoding_layer(self.FILTERS, False, dec6)
        dec7 = Concatenate(axis=-1)([dec7, enc1])

        # (256 x 256 x output_channel_count)
        dec8 = Activation(activation='relu')(dec7)
        dec8 = Conv2DTranspose(self.OUTPUT_CH, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(dec8)
        dec8 = Activation(activation='sigmoid')(dec8)

        return Model(inputs=inputs, outputs=dec8)

    def _add_encoding_layer(self, filters, sequence):
        new_sequence = LeakyReLU(0.2)(sequence)
        new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
        new_sequence = Conv2D(filters, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        return new_sequence

    def _add_decoding_layer(self, filters, add_drop_layer, sequence):
        new_sequence = Activation(activation='relu')(sequence)
        new_sequence = Conv2DTranspose(filters, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE,
                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        if add_drop_layer:
            new_sequence = Dropout(0.5)(new_sequence)
        return new_sequence

    def get_model(self):
        return self.UNET
