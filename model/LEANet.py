import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Dense, Flatten, Activation
from tensorflow.keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model

from model.component import conv_block
from model.CAAN_base import ModelWithAttentionBranchBase


class VGG16(ModelWithAttentionBranchBase):

    def __init__(self, img_shape=(256,256,3),
                       att_shape=(256,256,1),
                       att_points=[],
                       att_base_model="MobileNetV3",
                       input_method  = "none",
                       distil_method = "none",
                       sigmoid_apply = False, 
                       fusion_method = "none",
                       flat_method = "flat",
                       output_method = "separate"
                ):

        super(VGG16,self).__init__(att_shape,att_points,att_base_model,input_method,distil_method,sigmoid_apply,fusion_method,flat_method,output_method)
        self.img_shape = img_shape
        print(f"CAAN is created. Attention position is {att_points[0]+1}")
        
    def build(self):        
        
        n_filters=[64*1,64*2,64*4,64*8,64*8]
        
        input_ = Input(shape=self.img_shape, name="Main_Inputs")
        x = input_

        x = Conv2D(filters=n_filters[0], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = Conv2D(filters=n_filters[0], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
        x = self.BranchConnection(x,position=0)
        
        x = Conv2D(filters=n_filters[1], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = Conv2D(filters=n_filters[1], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
        x = self.BranchConnection(x,position=1)
        
        x = Conv2D(filters=n_filters[2], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = Conv2D(filters=n_filters[2], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = Conv2D(filters=n_filters[2], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
        x = self.BranchConnection(x,position=2)
        
        x = Conv2D(filters=n_filters[3], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = Conv2D(filters=n_filters[3], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = Conv2D(filters=n_filters[3], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
        x = self.BranchConnection(x,position=3)
        
        x = Conv2D(filters=n_filters[3], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = Conv2D(filters=n_filters[3], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = Conv2D(filters=n_filters[3], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
        x = self.BranchConnection(x,position=4)
        
        x = Flatten()(x)
        x = Dense(4096, activation="relu")(x)
        x = Dense(4096, activation="relu")(x)
        x = Dense(2, activation='softmax',name="main_pred")(x)
        pred = x
        
        inputs = [input_, self.att_input]

        if self.output_method=="separate":
            outputs = [pred, self.att_pred]

        elif self.output_method=="oneway":
            outputs = [pred]
        
        model = Model(inputs=inputs, outputs=outputs)
        
        return model    


class ResNet18(ModelWithAttentionBranchBase):

    def __init__(self, img_shape=(256,256,3),
                       att_shape=(256,256,1),
                       att_points=[],
                       att_base_model="MobileNetV3",
                       input_method  = "none",
                       distil_method = "none",
                       sigmoid_apply = False, 
                       fusion_method = "none",
                       flat_method = "flat",
                       output_method = "separate"
                ):
        super(ResNet18,self).__init__(att_shape,att_points,att_base_model,input_method,distil_method,sigmoid_apply,fusion_method,flat_method,output_method)
        self.img_shape = img_shape
        print(f"CAAN is created. Attention position is {att_points[0]+1}")
    
    def build(self):        
        
        num_filters = 64
        num_blocks = 4
        num_sub_blocks = 2

        input_ = Input(shape=self.img_shape, name="Main_Inputs")
        x = input_
        
        x = Conv2D(filters=num_filters, kernel_size=(7,7), padding='same', strides=2, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = self.BranchConnection(x,position=0) # 128 out
        x = Conv2D(num_filters, kernel_size=3, padding='same', strides=1, kernel_initializer='he_normal')(x)
        
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='block2_pool')(x)

        n_position=1
        for i in range(num_blocks):
            for j in range(num_sub_blocks):
                
                strides=1
                
                is_first_layer_but_not_first_block=False
                if j==0 and i>0:
                    is_first_layer_but_not_first_block=True
                    strides=2

                y = Conv2D(num_filters, kernel_size=3, padding='same', strides=strides, kernel_initializer='he_normal')(x)
                y = BatchNormalization()(y)
                y = Activation('relu')(y)
                y = Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(y)
                y = BatchNormalization()(y)
                
                # Skip structure
                if is_first_layer_but_not_first_block:
                    x = Conv2D(num_filters, kernel_size=1, padding='same', strides=2, kernel_initializer='he_normal')(x)
                
                x = Add()([x, y])
                x = Activation('relu')(x)

                # last of sub block 
                if j==num_sub_blocks-1:
                    x = self.BranchConnection(x,position=n_position) # 64,32,16,8 out
                    # x = Conv2D(num_filters, kernel_size=3, padding='same', strides=1, kernel_initializer='he_normal')(x)
                    n_position+=1

            num_filters *= 2

        x    = GlobalAveragePooling2D()(x)
        pred = Dense(2, activation='sigmoid',name="main_pred")(x)
        

        inputs = [input_, self.att_input]

        if self.output_method=="separate":
            outputs = [pred, self.att_pred]

        elif self.output_method=="oneway":
            outputs = [pred]
            
        model = Model(inputs=inputs, outputs=outputs)
        
        return model  


class CNN(ModelWithAttentionBranchBase):

    def __init__(self, img_shape=(256,256,3),
                       att_shape=(256,256,1),
                       att_points=[],
                       att_base_model="MobileNetV3",
                       input_method  = "none",
                       distil_method = "none",
                       sigmoid_apply = False, 
                       fusion_method = "none",
                       flat_method = "flat",
                       output_method = "separate"
                ):
        super(CNN,self).__init__(att_shape,att_points,att_base_model,input_method,distil_method,sigmoid_apply,fusion_method,flat_method,output_method)

        self.img_shape = img_shape
        self.filters = [64,64*2,64*4,64*8,64*8]
        print(f"CAAN is created. Attention position is {att_points[0]+1}")
    
    def build(self):    

        input_ = Input(shape=self.img_shape, name="Main_Inputs")
        x = input_

        for i in range(len(self.filters)):
            x = conv_block(self.filters[i],x)
            x = self.BranchConnection(x,position=i)

        x = Flatten()(x)
        pred = Dense(2, activation='sigmoid')(x)

        inputs=[input_,self.att_input]

        if self.output_method=="separate":
            outputs = [pred, self.att_pred]
        elif self.output_method=="oneway":
            outputs = [pred]
            
        model = Model(inputs=inputs, outputs=outputs)

        return model    