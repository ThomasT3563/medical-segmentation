#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf

class custom_Unet3D(object):
    
    """
        Upgrade of version My_UNet3d_V10 in order to reduce overfitting

        custom Unet implementation for multiclasse semantic segmentation
    """
    
    def __init__(self,shape_image,number_class):
        self.num_classes = number_class
        self.shape_image = shape_image
        
        #self.filters = [16,32,64,128,256]
        self.filters =[8,16,32,64,128]
        #self.filters =[4,8,16,32,64]
        self.kernel = (3,3,3)
        self.activation = tf.keras.layers.LeakyReLU()
        self.padding = "same" # peut etre passer a "valid" pour enlever le padding
        self.pooling = (2,2,2)
    
    def double_convolution(self,input_,num_stage):
        layer1 = tf.keras.layers.Conv3D(filters=self.filters[num_stage], 
                                      kernel_size=self.kernel,
                                      padding=self.padding
                                     )(input_)
        layer2 = self.activation(layer1)
        layer3 = tf.keras.layers.Conv3D(filters=self.filters[num_stage], 
                                      kernel_size=self.kernel,
                                      padding=self.padding
                                     )(layer2)
        layer4 = self.activation(layer3)
        layer5 = tf.keras.layers.SpatialDropout3D(0.2)(layer4) # adapatation in progress
        return layer5
    
        #tf.keras.layers.SpatialDropout3D
        #layer5 = tf.keras.layers.Dropout(0.1)(layer4)
        #return layer5
    
    def maxpooling(self,intput_):
        layer = tf.keras.layers.MaxPool3D(pool_size=self.pooling,
                                          padding=self.padding
                                         )(intput_)
        return layer
    
    def upsampling(self,input_):
        layer = tf.keras.layers.UpSampling3D(size=self.pooling
                                            )(input_)
        return layer
    
    def concatenate(self,upconv_input,forward_input):
        layer = tf.keras.layers.Concatenate()([upconv_input, forward_input])
        return layer
    
    def final_convolution(self,input_):
        layer = tf.keras.layers.Conv3D(filters=self.num_classes,
                                       kernel_size=(1, 1, 1),
                                       padding=self.padding
                                       )(input_)
        return layer

    def compression_block(self,input_,num_stage):
        """
            output : (forward_output,maxpooled_output)
        """
        layer1 = self.double_convolution(input_,num_stage)
        layer2 = self.maxpooling(layer1)
        layer3 = tf.keras.layers.BatchNormalization()(layer2)
        return (layer1,layer3)
    
    def bottleneck(self,input_,num_stage):
        layer1 = self.double_convolution(input_,num_stage)
        layer2 = tf.keras.layers.BatchNormalization()(layer1)
        return layer2
    
    def expansion_block(self,upconv_input,forward_input,num_stage):
        upconv_input = self.upsampling(upconv_input)
        layer1 = self.concatenate(upconv_input,forward_input)
        layer2 = self.double_convolution(layer1,num_stage)
        layer3 = tf.keras.layers.BatchNormalization()(layer2)
        return layer3
      
    def get_model(self):
        
        input_ = tf.keras.layers.Input(shape=self.shape_image,dtype=tf.float32,name="MODEL_INPUT")
        
        #compression
        (forward0,d_conv0) = self.compression_block(input_ ,num_stage=0)
        (forward1,d_conv1) = self.compression_block(d_conv0,num_stage=1)
        (forward2,d_conv2) = self.compression_block(d_conv1,num_stage=2)
        (forward3,d_conv3) = self.compression_block(d_conv2,num_stage=3)

        #bottleneck
        u_conv4 = self.bottleneck(d_conv3,num_stage=4)
        
        #expansion
        u_conv3 = self.expansion_block(u_conv4,forward3,num_stage=3)
        u_conv2 = self.expansion_block(u_conv3,forward2,num_stage=2)
        u_conv1 = self.expansion_block(u_conv2,forward1,num_stage=1)
        u_conv0 = self.expansion_block(u_conv1,forward0,num_stage=0)

        #last operations
        logits = self.final_convolution(u_conv0)
        output_ = tf.keras.layers.Softmax(name='MODEL_OUTPUT')(logits)
        model = tf.keras.models.Model(input_, output_,name='UNet_V13')
        
        return model

    
    
    