import os, datetime, random

from django.shortcuts import render #
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage #

import pickle as pkl
# from prepro import *
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from torch.autograd import Variable
from PIL import Image
from itsp.settings import MEDIA_ROOT

import json
from models import model
from models.data import Im2LatexDataset
# from models.build_vocab import Vocab
from models.utils import collate_fn
from models.model import Im2LatexModel
from models.decoding import LatexProducer
from models.score import score_files


from argparse import Namespace
args = Namespace(
    model_path = "ckpts/best_ckpt.pt",

    # model args
    beam_size = 5,
    result_path = "./results/result.txt",
    ref_path = "./results/ref.txt",
    split = "validate",

    # model args
    emb_dim = 80,
    dec_rnn_h = 512,
    data_path = "./data/",
    add_position_features = False,

    #training args
    max_len = 150,
    dropout = 0,
    cuda = False,
    batch_size = 8,
    epoches = 1,
    lr = 3e-4,
    min_lr = 3e-5,
    sample_method = "teacher_forcing", # Other opts: 'exp', 'inv_sigmoid'
    decay_k = 1. ,
    lr_decay = 0.5,
    lr_patience = 3,
    clip = 2.0,
    save_dir = "./ckpts",
    print_freq = 100,
    seed = 2020,
    from_check_point = False, 
    use_cuda = False,
)

class Vocab(object):
    def __init__(self):
        self.sign2id = {"<s>": START_TOKEN, "</s>": END_TOKEN,
                        "<pad>": PAD_TOKEN, "<unk>": UNK_TOKEN}
        self.id2sign = dict((idx, token)
                            for token, idx in self.sign2id.items())
        self.length = 4

    def add_sign(self, sign):
        if sign not in self.sign2id:
            self.sign2id[sign] = self.length
            self.id2sign[self.length] = sign
            self.length += 1

    def __len__(self):
        return self.length

with open('models/vocab.pkl', 'rb') as f:
	vocab = pkl.load(f)
mdl = Im2LatexModel(
        len(vocab), args.emb_dim, args.dec_rnn_h,
        add_pos_feat=args.add_position_features,
        dropout=
        args.dropout
    )



def index(request):
	context = {'a':1}
	return render(request, 'index.html', context)

latex_producer = LatexProducer(
    mdl, vocab, max_len=args.max_len,
    use_cuda=args.use_cuda, beam_size=args.beam_size)

def predictImage(request):
    print (request)
    print (request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    # print(filePathName)
    testimage='.'+filePathName

    img = Image.open(testimage)

    transform = transforms.ToTensor()
    img_tensor = transform(img)

    print(img_tensor.shape)
    img_tensor = img_tensor.unsqueeze(0)
    print(img_tensor.shape)

    # img_tensor = img_tensor.permute(0, 3, 1, 2) 
    result = latex_producer(img_tensor)
    # result = 5

    print(result)

    context={'filePathName':filePathName,'result':result}
    return render(request,'index.html',context) 

def viewDatabase(request):
    import os
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB.html', context) 

def viewAbout(request):
    context = {}
    return render(request, 'about.html', context)