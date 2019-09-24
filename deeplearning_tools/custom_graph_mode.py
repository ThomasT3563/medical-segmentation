#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import SimpleITK as sitk
import sys
            
class history_class(object):
    """
        Wrapper to allow compatibility with tensorflow fitter
    """
    def __init__(self,input_dict):
        self.history = input_dict

class model_trainer_machine(object):
    """
        Class to train the model defined in graph mode 
    """
    def __init__(self,model,loss_object,metrics,optimizer,buffer_size,train_dataset,valid_dataset,epochs=1):
        self.model         = model
        self.loss_object   = loss_object
        self.metrics       = metrics
        self.optimizer     = optimizer
        self.buffer_size   = buffer_size
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.epochs        = epochs
        self.history       = self.init_history()
        self.avg_loss      = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)

    def init_history(self):
        #training
        history_ = {'loss':[]}
        for metric in self.metrics:
            history_[metric.name] = []
        #validation
        history_['val_loss'] = []
        for metric in self.metrics:
            history_['val_'+metric.name] = []
        return history_
    
    def get_history(self):
        return history_class(self.history)
    
    @tf.function
    def train_step(self,scan, mask):
        with tf.GradientTape() as tape:
            # Make a prediction
            prediction = self.model(scan)
            # Get the error/loss using the loss_object previously defined
            loss = self.loss_object(mask, prediction)
        self.avg_loss.update_state(loss)
        # Compute the gradient which respect to the loss
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Change the weights of the model
        self.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))
        # The metrics are accumulate over time. You don't need to average it yourself.
        for metric in self.metrics: 
            metric(mask, prediction)
    
    @tf.function
    def valid_step(self,scan,mask):
        prediction = self.model(scan)
        self.avg_loss.update_state(self.loss_object(mask, prediction))
        for metric in self.metrics: 
            metric(mask, prediction)
    
    def fit(self):    
        
        for epoch in range(self.epochs):
            sys.stdout.write("Epoch "+str(epoch+1)+'/'+str(self.epochs)+"\n")
            
            # loop on batches of training data
            for idx,(scan,mask) in enumerate(self.train_dataset.shuffle(self.buffer_size)):
                #definition of loading bar
                length = round((idx+1)/self.buffer_size*20)
                loading_bar = "["+"="*length+">"+"-"*(20-length)+"]"
                sys.stdout.write("\r%s/%s %s" % (str(idx+1),str(self.buffer_size),loading_bar))
                self.train_step(scan, mask)
            
            # history update
            self.history['loss'].append(self.avg_loss.result().numpy())
            sys.stdout.write(" - loss: %0.6s" % (self.avg_loss.result().numpy()))
            self.avg_loss.reset_states()
            for metric in self.metrics:
                self.history[metric.name].append(metric.result().numpy())
                sys.stdout.write(" - %s: %0.6s" % (metric.name,metric.result().numpy()))
                metric.reset_states()
                
            # loop on batches of validation data
            for (scan,mask) in self.valid_dataset:
                self.valid_step(scan,mask)
            # history update
            self.history['val_loss'].append(self.avg_loss.result().numpy())
            sys.stdout.write(" - val_loss: %0.6s" % (self.avg_loss.result().numpy()))
            self.avg_loss.reset_states()
            for metric in self.metrics:
                self.history['val_'+metric.name].append(metric.result().numpy())
                sys.stdout.write(" - val_%s: %0.6s" % (metric.name,metric.result().numpy()))
                metric.reset_states()

            sys.stdout.write("\n")
        
        return history_class(self.history)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
