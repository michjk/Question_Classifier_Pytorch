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
    return len(ex.text)

class ModelRunner:
    def __init__(self, model_factory, loss_factory, optimizer_factory, epochs, batch_size, learning_logger, transpose = False, use_gpu = True):
        self.model_factory = model_factory
        self.loss_factory = loss_factory
        self.optimizer_factory = optimizer_factory
        self.epochs = epochs
        self.transpose = transpose
        self.learning_logger = learning_logger
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        
    def get_iterator(self, dataset, batch_size, train=True, shuffle=True, repeat=False, sort_key = sort_key):
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
        train_examples = train_data.examples
        train_fields = train_data.fields

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
        best_dev_acc = 0
        best_dev_loss = 0
        best_truth_res = []
        best_pred_res = []
        
        self.model = self.model_factory.create_class()
        
        update_parameter = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = self.optimizer_factory.create_class({'params':update_parameter})
        
        self.loss_function = self.loss_factory.create_class()

        train_iter = self.get_iterator(train_data, self.batch_size)
        dev_iter = self.get_iterator(dev_data, len(dev_data), train=False)

        self.learning_logger.initialize()

        for i in range(self.epochs):
            print('epoch: %d start!' % i)
            self.learn_epoch(train_iter, i)
            
            print('now best dev acc:',best_dev_acc)
            dev_acc, dev_loss, truth_res, pred_res = self.evaluate(dev_iter, i)
            
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_dev_loss = dev_loss
                print('New Best Dev!!!')
                self.learning_logger.save_model(self.model)
                best_truth_res = truth_res
                best_pred_res = pred_res
        
        self.learning_logger.save_confusion_matrix(best_truth_res, best_pred_res)
    
    def learn_cv(self, train_data, test_data, n_folds):
        best_val_acc_list = []
        best_val_loss_list = []

        train_val_generator = self.get_dataset_cv(train_data, n_folds=n_folds)
        
        for fold, (train_data_fold, val_data_fold) in enumerate(train_val_generator):
            self.model = self.model_factory.create_class()
            
            update_parameter = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = self.optimizer_factory.create_class({'params':update_parameter})
            
            self.loss_function = self.loss_factory.create_class()
            
            print("Fold: ", fold)
            best_val_acc = 0.0
            best_val_loss = 0.0
            
            best_truth_res = []
            best_pred_res = []

            train_iter = self.get_iterator(train_data_fold, self.batch_size)
            val_iter = self.get_iterator(val_data_fold, len(val_data_fold), train=False)
            for i in range(self.epochs):
                #print('epoch: %d start!' % i)
                self.learn_epoch(train_iter, i, cv=True)
                
                #print('now best dev acc:',best_val_acc)
                val_acc, val_loss, truth_res, pred_res = self.evaluate(val_iter, i, cv=True)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    #print('New Best Dev!!!')
                    best_truth_res = truth_res
                    best_pred_res = pred_res
            print("best val acc: ", best_val_acc)
            print("best val loss: ", best_val_loss)
            best_val_acc_list.append(best_val_acc)
            best_val_loss_list.append(best_val_loss)
        
        print("All cross validation accuracy: ", best_val_acc_list)
        print("Avg cross validation accuracy: ", np.average(best_val_acc_list))

        print("All cross validation loss: ", best_val_loss_list)
        print("Avg cross validation loss: ", np.average(best_val_loss_list))
        
        
    def learn_epoch(self, train_iter, i, cv = False):
        self.model.train()
        
        avg_loss = 0.0
        count = 0
        truth_res = []
        pred_res = []
        
        for batch in train_iter:
            sent, label = batch.text, batch.label
            if count == 0:
                print('train')
                print(sent)
            if self.transpose:
                sent.data.t_()
            label.data.sub_(1)
            truth_res += list(label.data)
            pred = self.model(sent)
            pred_label = pred.data.max(1)[1]
            pred_res += [x for x in pred_label]
            self.model.zero_grad()
            loss = self.loss_function(pred, label)
            avg_loss += loss.data[0]
            count += 1
            '''
            if count % 100 == 0:
                print('epoch: %d iterations: %d loss :%g' % (i, count*model.batch_size, loss.data[0]))
            '''
            loss.backward()
            self.optimizer.step()
        avg_loss /= len(train_iter)
        acc = self.get_accuracy(truth_res,pred_res)
        #print('epoch: %d done!\ntrain avg_loss:%g , acc:%g'%(i, avg_loss, acc))

        if not cv:
            self.learning_logger.train_log_value("accuracy", acc, i)
            self.learning_logger.train_log_value("loss", avg_loss, i)

    def evaluate(self, eval_iter, i, cv=False):
        self.model.eval()
        
        avg_loss = 0.0
        truth_res = []
        pred_res = []
        
        for batch in eval_iter:
            sent, label = batch.text, batch.label
            if self.transpose:
                sent.data.t_()
            label.data.sub_(1)
            truth_res += list(label.data)
            pred = self.model(sent)
            pred_label = pred.data.max(1)[1]
            pred_res += [x for x in pred_label]
            loss = self.loss_function(pred, label)
            avg_loss += loss.data[0]
        
        avg_loss /= len(eval_iter)
        acc = self.get_accuracy(truth_res, pred_res)
        print('dev avg_loss:%g train acc:%g' % (avg_loss, acc))
        if not cv:
            self.learning_logger.dev_log_value("accuracy", acc, i)
            self.learning_logger.dev_log_value("loss", avg_loss, i)

        return acc, avg_loss, truth_res, pred_res
    
    def get_accuracy(self, truth, pred):
        assert len(truth)==len(pred)
        right = 0
        for i in range(len(truth)):
            if truth[i]==pred[i]:
                right += 1.0
        return right/len(truth)
