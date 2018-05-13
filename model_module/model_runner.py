import os
from data_module.data_writer import * 
import shutil
import torch
from torchtext import data
from sklearn.model_selection import KFold
import numpy as np
import copy
import torch.autograd as autograd

def sort_key(ex):
    '''
        Function for sorting dataset based on length. It is applicable for sorting test dataset.

        Inputs:
            - ex (data.Example): one record of dataset
    '''
    return len(ex.text)

class ModelRunner:
    '''
        This is class for run training model and also cross validation model
    '''
    def __init__(self, model_factory, loss_factory, optimizer_factory, epochs, batch_size, learning_logger, use_gpu = True):
        '''
            Inputs:
                - model_factory (FactoryClass): a factory class for creating a model.
                - loss_factory (FactoryClass): a factory class for creating loss function.
                - optimizer_factory (FactoryClass): a factory class for creating optimizer function
                - epochs (int): number of epochs or rounds of training
                - batch_size (int): batch size of dataset to be passed to model in one foward pass
                - learning_logger: Class for writing logs and trained model
                - use_gpu (boolean): use gpu for training
        '''
        self.model_factory = model_factory
        self.loss_factory = loss_factory
        self.optimizer_factory = optimizer_factory
        self.epochs = epochs
        self.learning_logger = learning_logger
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        
    def get_iterator(self, dataset, batch_size, train=True, shuffle=True, repeat=False, sort_key = sort_key):
        '''
            Generate iterator from torchtext.data.Dataset

            Inputs:
                - dataset (data.Dataset): dataset instance
                - batch_size (int)
                - train (boolean): Is it train dataset or test dataset
                - shuffle (boolean): shuffle dataset or not
                - repeat (boolean): repeat iteration or not
                - sort_key (boolean): repeat dataset iteration or not
        '''
        device = -1
        if self.use_gpu:
            device = None
        
        if train:
            sort_key = None

        dataset_iter = data.Iterator(
            dataset, batch_size=batch_size, device=device,
            train=train, shuffle=shuffle, repeat=repeat, sort_key=sort_key
        )

        return dataset_iter

    def get_dataset_cv(self, train_data, n_folds = 10):
        '''
            Generate n_folds set iterator from torchtext.data.Dataset for cross validation

            Inputs:
                - train_data (data.Dataset): train dataset instance
                - n_folds (int): number of folds
        '''
        train_examples = train_data.examples
        train_fields = train_data.fields
        
        # create folds
        def iter_folds():
            train_examples_np = np.array(train_examples)
            kf = KFold(n_splits=n_folds)
            for train_idx, val_idx in kf.split(train_examples_np):
                yield (
                    data.Dataset(train_examples_np[train_idx], train_fields),
                    data.Dataset(train_examples_np[val_idx], train_fields)
                )
        
        return iter_folds()
    
    def learn(self, train_data, dev_data):
        '''
            Start train model

            Inputs:
                - train_data (data.Dataset): training dataset
                - dev_data (data.Dataset): test dataset
        '''
        best_dev_acc = 0
        best_dev_loss = None
        best_truth_res = []
        best_pred_res = []
        
        # create model instance
        self.model = self.model_factory.create_class()
        
        # create optmizer instance
        update_parameter = filter(lambda p: p.requires_grad, self.model.parameters()) # only update parameter that indicates need updates
        self.optimizer = self.optimizer_factory.create_class({'params':update_parameter})
        
        # create loss function
        self.loss_function = self.loss_factory.create_class()

        # generate iterator
        train_iter = self.get_iterator(train_data, self.batch_size)
        dev_iter = self.get_iterator(dev_data, len(dev_data), train=False)

        # initialize folder
        self.learning_logger.initialize()

        # start training
        for i in range(self.epochs):
            # train
            print('epoch: %d start!' % i)
            self.learn_epoch(train_iter, i)
            
            # test
            print('now best dev acc:',best_dev_acc)
            dev_acc, dev_loss, truth_res, pred_res = self.evaluate(dev_iter, i)
            
            # save best model up to current epoch
            if best_dev_loss is None or dev_loss < best_dev_loss:
                best_dev_acc = dev_acc
                best_dev_loss = dev_loss
                print('New Best Dev!!!')
                self.learning_logger.save_model(self.model)
                best_truth_res = truth_res
                best_pred_res = pred_res
        
        print("best model accuracy: ", best_dev_acc)
        print("best model error: ", best_dev_loss)
        self.learning_logger.save_confusion_matrix(best_truth_res, best_pred_res)
    
    def learn_cv(self, train_data, n_folds):
        '''
            Start cross validate model

            Inputs:
                - train_data (data.Dataset): training dataset
                - n_folds (int): number of folds
        '''
        best_val_acc_list = []
        best_val_loss_list = []

        # genereate iterators for each folds
        train_val_generator = self.get_dataset_cv(train_data, n_folds=n_folds)
        
        # start cross validation
        for fold, (train_data_fold, val_data_fold) in enumerate(train_val_generator):
            
            # create model
            self.model = self.model_factory.create_class()
            
            # create optimizer
            update_parameter = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = self.optimizer_factory.create_class({'params':update_parameter})
            
            # create loss function
            self.loss_function = self.loss_factory.create_class()
            
            print("Fold: ", fold)
            best_val_acc = 0.0
            best_val_loss = None
            
            best_truth_res = []
            best_pred_res = []

            # create data iterator
            train_iter = self.get_iterator(train_data_fold, self.batch_size)
            val_iter = self.get_iterator(val_data_fold, len(val_data_fold), train=False)
            for i in range(self.epochs):
                # train
                print('epoch: %d start!' % i)
                self.learn_epoch(train_iter, i, cv=True)
                
                # test
                print('now best dev acc:',best_val_acc)
                val_acc, val_loss, truth_res, pred_res = self.evaluate(val_iter, i, cv=True)
                
                # best model from entire epoch
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    print('New Best Dev!!!')
                    best_truth_res = truth_res
                    best_pred_res = pred_res
            print("best val acc: ", best_val_acc)
            print("best val loss: ", best_val_loss)
            best_val_acc_list.append(best_val_acc)
            best_val_loss_list.append(best_val_loss)
        
        # output result
        print("All cross validation accuracy: ", best_val_acc_list)
        print("Avg cross validation accuracy: ", np.average(best_val_acc_list))

        print("All cross validation loss: ", best_val_loss_list)
        print("Avg cross validation loss: ", np.average(best_val_loss_list))
        
        
    def learn_epoch(self, train_iter, i, cv = False):
        '''
            Train in one epoch

            Inputs:
                - train_iter (iterator): Iterator train dataset
                - i (int): i-th epoch
                - cv (boolean): cross validation process or not
            
        '''
        self.model.train()
        
        avg_loss = 0.0
        count = 0
        truth_res = []
        pred_res = []
        
        # pass each batch to model
        for batch in train_iter:
            sent, label = batch.text, batch.label
            truth_res += [int(x) for x in label.data]
            pred = self.model(sent)
            pred_label = pred.data.max(1)[1]
            pred_res += [int(x) for x in pred_label]
            self.model.zero_grad()
            loss = self.loss_function(pred, label)
            avg_loss += float(loss.data[0])
            count += 1
            
            if count % 100 == 0:
                print('epoch: %d iterations: %d loss :%g' % (i, count*model.batch_size, loss.data[0]))
            
            loss.backward()
            self.optimizer.step()
        
        # calculate result
        avg_loss /= len(train_iter)
        acc = self.get_accuracy(truth_res,pred_res)
        print('epoch: %d done!\ntrain avg_loss:%g , acc:%g'%(i, avg_loss, acc))

        # not support saving accuracy & loss for cross validation 
        if not cv:
            self.learning_logger.train_log_value("accuracy", acc, i)
            self.learning_logger.train_log_value("loss", avg_loss, i)

    def evaluate(self, eval_iter, i, cv=False):
        '''
            Evaluate model in one epoch
        
            Inputs:
                - eval_iter (iterator): test/eval dataset iterator
                - i (int): i-th epoch
                - cv : cross validation or not
            
            Outputs:
                - acc (double): accuracy
                - avg_loss (double): average loss of current trained model
                - truth_res (list): truth label
                - pred_res (list): predicted label
        '''
        self.model.eval()
        
        avg_loss = 0.0
        truth_res = []
        pred_res = []
        
        # pass each batch to model
        for batch in eval_iter:
            sent, label = batch.text, batch.label
            truth_res += [int(x) for x in label.data]
            pred = self.model(sent)
            pred_label = pred.data.max(1)[1]
            pred_res += [int(x) for x in pred_label]
            loss = self.loss_function(pred, label)
            avg_loss += float(loss.data[0])
        
        #calculate result
        avg_loss /= len(eval_iter)
        acc = self.get_accuracy(truth_res, pred_res)
        print('dev avg_loss:%g train acc:%g' % (avg_loss, acc))
        
        # not support saving accuracy & loss for cross validation
        if not cv:
            self.learning_logger.dev_log_value("accuracy", acc, i)
            self.learning_logger.dev_log_value("loss", avg_loss, i)

        return acc, avg_loss, truth_res, pred_res
    
    def get_accuracy(self, truth, pred):
        '''
            Calculate percentage accuracy

            Inputs:
                - truth (list): list of truth label
                - pre (list): list of predicted label
            
            Outputs:
                - accuracy in percentage
        '''
        assert len(truth)==len(pred)
        right = 0
        for i in range(len(truth)):
            if truth[i]==pred[i]:
                right += 1.0
        return right/len(truth)
