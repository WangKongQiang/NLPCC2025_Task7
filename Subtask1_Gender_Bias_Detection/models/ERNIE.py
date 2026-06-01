import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from transformers import RobertaTokenizer, RobertaModel
class Config(object):

    def __init__(self, dataset):
        self.model_name = 'ERNIE'
        self.train_path = dataset + '/train.txt'
        self.dev_path = dataset + '/dev.txt'
        self.test_path = dataset + '/test.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/class.txt').readlines()]
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 5
        self.batch_size = 32
        self.pad_size = 64
        self.learning_rate = 1e-5
        self.bert_path_2 = './models--nghuyong--ernie-1.0-base-zh'
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path_2)
        self.hidden_size_2 = 768


class Model(nn.Module):

    def __init__(self, config):
        
        super(Model, self).__init__()
        self.bert = AutoModel.from_pretrained(config.bert_path_2, return_dict=False)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size_2, config.num_classes)

    def forward(self, x):
        
        context = x[0]
        mask = x[2]
        _,pooled = self.bert(context, attention_mask=mask)
        out = self.fc(pooled)
        return out