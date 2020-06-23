from os.path import join
import os, datetime, random

from functools import partial
from django.views import View
from django.shortcuts import render #
from torch.utils.data import DataLoader
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage #
from django_tex.core import compile_template_to_pdf
from django_tex.shortcuts import render_to_pdf

import cv2
import PIL
import numpy as np 
import matplotlib.pyplot as plt 

import reportlab
import pickle as pkl
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from torch.autograd import Variable
from itsp.settings import MEDIA_ROOT

import json
from models import model
from models.data import Im2LatexDataset
from models.build_vocab import Vocab, load_vocab
from models.utils import collate_fn
from models.model.model import Im2LatexModel
from models.model.decoding import LatexProducer
from models.model.score import score_files
from models.inputImage import preImage


from argparse import Namespace





# class Im2LatexModel(View):
    # def __init__ ():
args = Namespace(
    model_path = "models/data/best_ckpt.pt",

    # model args
    beam_size = 5,
    
    # model args
    emb_dim = 80,
    dec_rnn_h = 512,
    data_path = "models/data/",
    add_position_features = False,

    #training args
    max_len = 150,
    dropout = 0,
    cuda = False,
    epoches = 15,
    lr = 3e-4,
    min_lr = 3e-5,
    batch_size=8,
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
# with open('models/data/vocab.pkl', 'rb') as f:
# 	vocab = pkl.load(f)
vocab = load_vocab(args.data_path)
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

latex_producer = LatexProducer(
    model, vocab, max_len=args.max_len,
    use_cuda=args.use_cuda, beam_size=args.beam_size)






def predict(request):
    # doc = request.FILES #returns a dict-like object
    # fileObj = doc['filename']

    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName

    img = Image.open(testimage)
    img = img.convert(mode='RGB')

    trans = transforms.ToTensor()
    img = trans(img)

    img = img.unsqueeze(0)
    print(img.shape)

    result = latex_producer(img)

    result = '\n'.join(result)

    template = 'test.tex'
    contextPdf = {'equation': result}
    PDF = compile_template_to_pdf(template, contextPdf)

    f = open('media/pdf/prediction.pdf', 'w+b')
    f.write(PDF)
    f.close()

    context={'filePathName':filePathName,'result':result}
    return render(request,'index.html',context)




def viewDatabase(request):
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB.html', context) 




def index(request):
	context = {'a':1}
	return render(request, 'index.html', context)




def viewAbout(request):
    context = {}
    return render(request, 'about.html', context)





