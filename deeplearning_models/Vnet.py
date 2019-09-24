#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf

class My_VNet_V1(object):
    
    def __init__(self,SHAPE_IMAGE):
        """
        
        """
        self.num_classes = 5
        self.shape_image = SHAPE_IMAGE

        self.filters = [8,16,32,64,128,256]
        self.kernel_conv = (3,3,3)
        self.stride_conv = (1,1,1)
        self.activation = tf.keras.layers.LeakyReLU() #tf.keras.layers.PReLU()
        self.padding = "same"
        
        self.kernel_pool = (2,2,2)
        self.stride_pool = (2,2,2)
    
    def convolution(self,INPUT,num_stage):
        """
            num_layer : (int) number of the current stage in the CNN (max=4)
        """
        layer1 = tf.keras.layers.Conv3D(filters=self.filters[num_stage],
                                        kernel_size=self.kernel_conv,
                                        padding=self.padding,
                                        strides=self.stride_conv,
                                       )(INPUT)
        layer2 = self.activation(layer1)
        return layer2
    
    def final_convolution(self,INPUT):
        """
            used at last operation in vnet
        """
        layer1 = tf.keras.layers.Conv3D(filters=self.num_classes,
                                        kernel_size=(1, 1, 1),
                                        padding=self.padding,
                                        strides=self.stride_conv,
                                       )(INPUT)
        layer2 = self.activation(layer1)
        return layer2
    
    def residual_learning(self,stage_input,conv_output):
        """
            stage_input : image at the beginning of the current stage
            conv_output : output from the last conv layer at the current stage
        """

        residual_connection = conv_output + tf.pad(stage_input,[[0,0],[0,0],[0,0],[0,0],[0,conv_output.shape[4]-stage_input.shape[4]]])
        return residual_connection
    
    def down_conv(self,INPUT,num_stage):
        """
        """
        layer1 = tf.keras.layers.Conv3D(filters=self.filters[num_stage],
                                        kernel_size=self.kernel_pool,
                                        padding=self.padding,
                                        strides=self.stride_pool,
                                       )(INPUT)
        return layer1
        
    def up_conv(self,INPUT,num_stage):
        """
        """
        layer1 = tf.keras.layers.Conv3DTranspose(filters=self.filters[num_stage],
                                                 kernel_size=self.kernel_pool,
                                                 padding=self.padding,
                                                 strides=self.stride_pool,
                                                )(INPUT)
        return layer1
    
    def concatenate(self,upconv_input,forward_input):
        """
        """
        layer1 = tf.keras.layers.Concatenate()([upconv_input, forward_input])
        return layer1
    
    def compression_block(self,INPUT,num_stage,nb_conv):
        """
            get input layer, apply nb_conv convolutions and return down convoluted layer
            output : (forwarded features,down-convolved features)
        """
        layer_ = INPUT
        for _ in range(nb_conv):
            layer_ = self.convolution(layer_,num_stage)
        layer2 = self.residual_learning(INPUT,layer_)
        layer3 = self.down_conv(layer2,num_stage)
        layer4 = self.activation(layer3)
        return (layer2,layer4)
    
    def expansion_block(self,upconv_input,forward_input,num_stage,nb_conv):
        """
        """
        layer_ = self.concatenate(upconv_input,forward_input)
        for _ in range(nb_conv):
            layer_ = self.convolution(layer_,num_stage)   
        layer2 = self.residual_learning(layer_,upconv_input)
        layer3 = self.up_conv(layer2,(num_stage+1))
        layer4 = self.activation(layer3)
        return layer4
    
    def bottleneck(self,INPUT,num_stage=4):
        """
        """
        layer1 = self.convolution(INPUT,num_stage)
        layer2 = self.convolution(layer1,num_stage)
        layer3 = self.convolution(layer2,num_stage)
        layer4 = self.residual_learning(INPUT,layer3)
        layer5 = self.up_conv(layer4,num_stage)
        layer6 = self.activation(layer5)
        return layer6
        
    def get_model(self):
        """
        """
        
        input_ = tf.keras.layers.Input(shape=self.shape_image,dtype=tf.float32,name="MODEL_INPUT")
        
        #compression
        (forward0,d_conv0) = self.compression_block(input_, num_stage=0,nb_conv=1)
        (forward1,d_conv1) = self.compression_block(d_conv0,num_stage=1,nb_conv=2)
        (forward2,d_conv2) = self.compression_block(d_conv1,num_stage=2,nb_conv=3)
        (forward3,d_conv3) = self.compression_block(d_conv2,num_stage=3,nb_conv=3)
        
        #bottleneck
        u_conv4 = self.bottleneck(d_conv3) #num_stage=4
        
        #expansion
        u_conv3 = self.expansion_block(u_conv4,forward3,num_stage=4,nb_conv=3)
        u_conv2 = self.expansion_block(u_conv3,forward2,num_stage=3,nb_conv=3)
        u_conv1 = self.expansion_block(u_conv2,forward1,num_stage=2,nb_conv=2)
        
        #last operations
        conc0 = self.concatenate(u_conv1,forward0)
        conv0 = self.convolution(conc0,num_stage=1)
        resl0 = self.residual_learning(conv0,u_conv1)
        
        logits = self.final_convolution(resl0)
        dropout = tf.keras.layers.Dropout(0.5)(logits)
        output_ = tf.keras.layers.Softmax(name='MODEL_OUTPUT')(dropout)
        model = tf.keras.models.Model(input_, output_,name='VNet')
        
        return model
    