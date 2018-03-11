import os
from data_module.data_writer import * 
import shutil
import torch

class ModelRunner:
    def __init__(self, model, epochs, loss_function, optimizer, learning_logger, transpose = False):
        self.model = model
        self.epochs = epochs
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.transpose = transpose
        self.learning_logger = learning_logger

    def learn(self, train_iter, dev_iter):
        best_dev_acc = 0

        self.learning_logger.initialize()
        for i in range(self.epochs):
            print('epoch: %d start!' % i)
            self.learn_epoch(train_iter, i)
            
            print('now best dev acc:',best_dev_acc)
            dev_acc = self.evaluate(dev_iter, i)
            
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                """ 
                if os.path.exists(self.saved_model_file_path):
                    os.remove(self.saved_model_file_path)
                """
                print('New Best Dev!!!')
                self.learning_logger.save_model(self.model)
    
    def learn_epoch(self, train_iter, i):
        self.model.train()
        
        avg_loss = 0.0
        count = 0
        truth_res = []
        pred_res = []
        
        for batch in train_iter:
            sent, label = batch.text, batch.label
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
            if count % 100 == 0:
                print('epoch: %d iterations: %d loss :%g' % (i, count*model.batch_size, loss.data[0]))
            loss.backward()
            self.optimizer.step()
        avg_loss /= len(train_iter)
        acc = self.get_accuracy(truth_res,pred_res)
        print('epoch: %d done!\ntrain avg_loss:%g , acc:%g'%(i, avg_loss, acc))
        self.learning_logger.train_log_value("accuracy", acc, i)
        self.learning_logger.train_log_value("loss", avg_loss, i)

    def evaluate(self, eval_iter, i):
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
        self.learning_logger.dev_log_value("accuracy", acc, i)
        self.learning_logger.dev_log_value("loss", avg_loss, i)
        return acc
    
    def get_accuracy(self, truth, pred):
        assert len(truth)==len(pred)
        right = 0
        for i in range(len(truth)):
            if truth[i]==pred[i]:
                right += 1.0
        return right/len(truth)
