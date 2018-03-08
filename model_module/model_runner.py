import os
from data_module.data_output import * 
import shutil
import torch

class ModelRunner:
    def __init__(self, model, epochs, loss_function, optimizer, transpose = False, saved_model_folder_path = 'best_model', saved_model_name = 'best_model.model', log_folder_path = 'log', train_log_folder_name='train_log', dev_log_folder_name='test_log'):
        self.model = model
        self.epochs = epochs
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.transpose = transpose

        self.train_log_folder_path = os.path.join(log_folder_path, train_log_folder_name)
        self.dev_log_folder_path = os.path.join(log_folder_path, dev_log_folder_name)
        
        self.saved_model_folder_path = saved_model_folder_path
        self.saved_model_file_path = os.path.join(saved_model_folder_path, saved_model_name)

    def learn(self, train_iter, dev_iter):
        best_dev_acc = 0
        
        if os.path.exists(self.saved_model_folder_path):
            shutil.rmtree(self.saved_model_folder_path)
        os.makedirs(self.saved_model_folder_path)
        
        if os.path.exists(self.train_log_folder_path):
            shutil.rmtree(self.train_log_folder_path)
        if os.path.exists(self.dev_log_folder_path):
            shutil.rmtree(self.dev_log_folder_path)
        print(self.train_log_folder_path)
        print(self.dev_log_folder_path)
        train_logger = PlotLogger(self.train_log_folder_path)
        dev_logger = PlotLogger(self.dev_log_folder_path)

        for i in range(self.epochs):
            print('epoch: %d start!' % i)
            self.learn_epoch(train_iter, i, train_logger)
            
            print('now best dev acc:',best_dev_acc)
            dev_acc = self.evaluate(dev_iter, i, dev_logger)
            
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                if os.path.exists(self.saved_model_file_path):
                    os.remove(self.saved_model_file_path)
                print('New Best Dev!!!')
                torch.save(self.model.state_dict(), self.saved_model_file_path)
    
    def learn_epoch(self, train_iter, i, logger):
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
        logger.log_value("accuracy", acc, i)
        logger.log_value("loss", avg_loss, i)

    def evaluate(self, eval_iter, i, logger):
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
        logger.log_value("accuracy", acc, i)
        logger.log_value("loss", avg_loss, i)
        return acc
    
    def get_accuracy(self, truth, pred):
        assert len(truth)==len(pred)
        right = 0
        for i in range(len(truth)):
            if truth[i]==pred[i]:
                right += 1.0
        return right/len(truth)
