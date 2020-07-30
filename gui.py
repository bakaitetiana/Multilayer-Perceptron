# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 21:36:44 2019

@author: Tanya
"""

# -*- coding: utf-8 -*-
from tkinter import *
import logic
import matplotlib.pyplot as plt
import numpy as np


    
def explore_nn(input_l,target_l, n, verbose=False):
        total = 0
        errors = 0
        for i in range(len(input_l)):
            line = input_l[i]
            predictions = n.predict(line)
            ca = np.array(target_l[i]).argmax()
            p = predictions.argmax()
            total += 1
            if ca != p:
                errors += 1
        return 1 - errors / total # accuracy

class App(Frame):

    def __init__(self, master):
        super(App, self).__init__(master)
        self.grid()
        self.create_widgets()
    
    def create_widgets(self):
        Label(self, text = "Choose dataset:").grid(row = 0, column = 0)
        self.var = IntVar()
        self.var.set(0)
        self.file = Radiobutton(self, text = "CNAE-9", variable=self.var, value=0)
        self.file.grid(columnspan = 2)
        self.gui = Radiobutton(self, text = "SPECT", width=3, height=3, variable=self.var, value=1) 
        self.gui.grid(columnspan = 2)
        self.gui2 = Radiobutton(self, text = "haberman", variable=self.var, value=2) 
        self.gui2.grid(columnspan = 2)
        Label(self, text = "Choose number of hidden layers:").grid(row = 4, column = 0)
        self.v = IntVar()
        self.v.set(0)
        self.file = Radiobutton(self, text = "1", variable=self.var, value=0)
        self.file.grid(columnspan = 2)
        self.gui4 = Radiobutton(self, text = "2", width=3, height=3, variable=self.var, value=1) 
        self.gui4.grid(columnspan = 2)
        Label(self, text = "Choose number of neurons in hidden layers:").grid(row = 7, column = 0)
        Label(self, text = "1").grid(row = 8, column = 0)
        self.user_input = Entry(self)
        self.user_input.grid(row = 8, column = 1, sticky = W)
        Label(self, text = "2").grid(row = 9, column = 0) 
        self.us_input = Entry(self)
        self.us_input.grid(row = 9, column = 1, sticky = W)
        self.b2 = Button(self, text="Work", command= self.clicked)
        self.b2.grid(row = 13, column = 1)
        #self.b3 = Button(self, text = 'Exit', command = self.quit)
        #self.b3.grid(row = 14, column = 1)
        Label(self, text = "Accuracy").grid(row = 16, column = 0) 
        self.c = Text(self, width=30, height=8, wrap = WORD)
        self.c.grid(row = 16, column = 1)
        
        
    # training  
    def clicked(self):
        #p = self.user_input.get()
        if (self.var.get() == 0):
            p = self.user_input.get()
            p = int(p)
            f = open("C:/AI2/NN/CNAE-9.data","r")
            fl = f.readlines()
            all_input = []
            all_target = []
            for x in fl:
                all_input.append(list(map(int,x.split(',')[1:])))
                exptd = [0 for i in range(9)]
                exptd[int(x.split(',')[0])-1] = 1
                all_target.append(exptd)
            input_list = []
            target_list = []
            error_rate = []
            model = logic.NeuralNetwork(input_nodes=856, hidden_nodes=p, output_nodes=9)
            for x in fl:
                input_list.append(list(map(int,x.split(',')[1:])))
                exptd = [0 for i in range(9)]
                exptd[int(x.split(',')[0])-1] = 1
                target_list.append(exptd)
                error_rate.append(model.train(input_list[-1],target_list[-1],all_input,all_target,1000))
            res = explore_nn(input_list,target_list, model)
            #y, x = (input_list,target_list, model)
            plt.plot(range(len(input_list)), error_rate) 
            plt.xlabel('time') 
            plt.ylabel('error') 
            plt.title('Loss curve!') 
            plt.show() 
        elif (self.var.get() == 1):
            p = self.user_input.get()
            p = int(p)
            f = open("C:/AI2/NN/SPECT.train","r")
            fl = f.readlines()
            all_input = []
            all_target = []
            for x in fl:
                all_input.append(list(map(int,x.split(',')[1:])))
                exptd = [0 for i in range(2)]
                exptd[int(x.split(',')[0])] = 1
                all_target.append(exptd)
            input_list = []
            target_list = []
            error_rate = []
            model = NeuralNetwork(input_nodes=22, hidden_nodes=p, output_nodes=2)
            for x in fl:
                input_list.append(list(map(int,x.split(',')[1:])))
                exptd = [0 for i in range(2)]
                exptd[int(x.split(',')[0])] = 1
                target_list.append(exptd)
                error_rate.append(model.train(input_list,target_list,all_input,all_target,1000000000))
            res = explore_nn(input_list,target_list, model)
            #y, x = expl(input_list,target_list, model)
            plt.plot(range(len(input_list)), error_rate) 
            plt.xlabel('time') 
            plt.ylabel('error') 
            plt.title('Loss curve!') 
            plt.show() 
            #loss_value = expl(input_list,target_list, model)
        elif (self.var.get() == 2):
            p = self.user_input.get()
            p = int(p)
            f = open("C:/AI2/NN/haberman.data","r")
            fl = f.readlines()
            all_input = []
            all_target = []
            for x in fl:
                all_input.append(list(map(int,x.split(',')[:-1])))
                exptd = [0 for i in range(2)]
                exptd[int(x.split(',')[-1])-1] = 1
                all_target.append(exptd)            
            input_list = []
            target_list = []
            error_rate = []
            model = logic.NeuralNetwork(input_nodes=3, hidden_nodes=p,output_nodes=2)
            for x in fl:
                input_list.append(list(map(int,x.split(',')[:-1])))
                exptd = [0 for i in range(2)]
                exptd[int(x.split(',')[-1])-1] = 1
                target_list.append(exptd)
                error_rate.append(model.train(input_list,target_list,all_input,all_target,1))
            res = explore_nn(input_list,target_list, model)
            #y, x = expl(input_list,target_list, model)
            plt.plot(range(len(input_list)), error_rate) 
            plt.xlabel('time') 
            plt.ylabel('error') 
            plt.title('Loss curve!') 
            plt.show() 
        else:
            res = None
        self.c.delete('1.0',END)
        self.c.insert(END, res)


root = Tk()
root.title('GUI')
root.geometry('550x550')
app = App(root)
root.mainloop()



  
