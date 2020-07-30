# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 21:35:49 2019

@author: Tanya
"""

import os
import pickle
import errno

import numpy as np


STORAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.storage'))

def expl(input_l,target_l, n, verbose=False):
        total = 0
        errors = 0
        #er_list=[]
       # m = []
        for i in range(len(input_l)):
            line = input_l[i]
            predictions = n.predict(line)
            ca = np.array(target_l[i]).argmax()
            p = predictions.argmax()
            total += 1
            if ca != p:
                errors += 1
                #er = p - ca
                #er_list.append(er)
        return errors/total
       # return errors / total # accuracy

class NeuralNetwork:
    def __init__(self, input_nodes=3, hidden_nodes=3, output_nodes=3, rate=0.3, load_from=None):
        if load_from is not None:
            if not self.load(load_from):
                raise ValueError('Model with name `{}` not found'.format(load_from))
        else:
            self.input_nodes = input_nodes
            self.hidden_nodes = hidden_nodes
            self.output_nodes = output_nodes
            self.rate = rate
            self.w_i_h = None  
            self.w_h_o = None  
            self.__init_weights()

    def train(self, input_list, target_list, all_input, all_target,iterations):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T
        for i in range(iterations):


        
            h_inputs = np.dot(self.w_i_h, inputs)

        
            h_outputs = self.__activation_function(h_inputs)

        
            o_inputs = np.dot(self.w_h_o, h_outputs)

        
            o_outputs = self.__activation_function(o_inputs)

        
            o_errors = targets - o_outputs

            h_errors = np.dot(self.w_h_o.T, o_errors)

        
            self.w_h_o += self.rate * np.dot((o_errors * o_outputs * (1 - o_outputs)), h_outputs.T)

            self.w_i_h += self.rate * np.dot((h_errors * h_outputs * (1 - h_outputs)), inputs.T)
            
            return expl(all_input,all_target, self)

    def predict(self, input_list):
       
        inputs = np.array(input_list, ndmin=2).T

        h_inputs = np.dot(self.w_i_h, inputs)

       
        h_outputs = self.__activation_function(h_inputs)

       
        o_inputs = np.dot(self.w_h_o, h_outputs)

       
        o_outputs = self.__activation_function(o_inputs)

        return o_outputs

    def back_query(self, output_list):
       
        o_outputs = np.array(output_list, ndmin=2).T

        
        o_inputs = self.__back_activation_function(o_outputs)

        
        h_outputs = np.dot(self.w_h_o.T, o_inputs)

       
        h_outputs -= np.min(h_outputs)
        h_outputs /= np.max(h_outputs)
        h_outputs *= 0.98
        h_outputs += 0.01

       
        hidden_inputs = self.__back_activation_function(h_outputs)

        
        inputs = np.dot(self.w_i_h.T, hidden_inputs)

      
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs
    
   
        #return errors / total

    def __init_weights(self):
        self.w_i_h = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.w_h_o = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

    @staticmethod
    def __activation_function(x):
        return 1.0 / (1.0 + np.exp(-x))  

    @staticmethod
    def __back_activation_function(y):
        return np.log(y / (1.0 - y))
  
