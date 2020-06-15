from os.path import join
import os, datetime, random

from django.shortcuts import render #
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage #

import pickle as pkl
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
from models.build_vocab import Vocab
from models.utils import collate_fn
from models.model.model import Im2LatexModel
from models.model.decoding import LatexProducer
from models.model.score import score_files


from argparse import Namespace

args = Namespace(
    model_path = "models/data/best_ckpt.pt",

    # model args
    beam_size = 5,
    
    # model args
    emb_dim = 80,
    dec_rnn_h = 512,
    data_path = "./data/",
    add_position_features = False,

    #training args
    max_len = 150,
    dropout = 0,
    cuda = False,
    epoches = 15,
    lr = 3e-4,
    min_lr = 3e-5,
    # sample_method = "teacher_forcing", # Other opts: 'exp', 'inv_sigmoid'
    # decay_k = 1. ,
    
    # clip = 2.0,
    # save_dir = "./ckpts",
    # print_freq = 100,
    # seed = 2020,
    # from_check_point = False, 
    use_cuda = False,
)

print("Loading vocab...")
with open('models/data/vocab.pkl', 'rb') as f:
	vocab = pkl.load(f)
print("Vocab loaded!")


checkpoint = torch.load(join(args.model_path), map_location={'cuda:0': 'cpu'})
model_args = checkpoint['args']
print(model_args)
# print(checkpoint['model_state_dict'])

print("Loading model...")
model = Im2LatexModel(
        len(vocab), model_args.emb_dim, model_args.dec_rnn_h,
        add_pos_feat=model_args.add_position_features,
        dropout=model_args.dropout
    )
model.load_state_dict(checkpoint['model_state_dict'])
print("Model loaded!")

def index(request):
	context = {'a':1}
	return render(request, 'index.html', context)

latex_producer = LatexProducer(
    model, vocab, max_len=args.max_len,
    use_cuda=args.use_cuda, beam_size=args.beam_size)

def predictImage(request):
    # print (request)
    # print (request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName

    img = Image.open(testimage)
    img = img.convert(mode='RGB')

    trans = transforms.ToTensor()
    img = trans(img)

    print(img.shape)
    img = img.unsqueeze(0)
    # print(img_tensor.shape)

    # img_tensor = img_tensor.permute(0, 3, 1, 2) 
    result = latex_producer(img)
    # result = 5

    # print(result)

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