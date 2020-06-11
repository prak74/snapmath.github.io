import os, datetime, random

from django.shortcuts import render #
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage #

from prepro import *
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


img_height, img_widht = 224, 224

import pickle as pkl


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
    cuda = True,
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
    use_cuda = True,
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
print(vocab)

from test import vcb

# vocab = load_vocab('./models/')
mdl = Im2LatexModel(
        len(vocab), args.emb_dim, args.dec_rnn_h,
        add_pos_feat=args.add_position_features,
        dropout=
        args.dropout
    )

# with mdl.as_default():
#     pt_session = Session()
#     with pt_session.as_default():
#         model=torch.load('./models/best_ckpt.pt')

# Create your views here.

def index(request):
	context = {'a':1}
	return render(request, 'index.html', context)

latex_producer = LatexProducer(
    model, vocab, max_len=args.max_len,
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
    # result = 5
    result = latex_producer(img)
    # x = image.img_to_array(img)
    # x=x/255
    # x=x.reshape(1,img_height, img_width,3)
    
    # with mdl.as_default():
    #     with pt_session.as_default():
    #         predi=model.predict(x)

    # for imgs, tgt4training, tgt4cal_loss in tqdm(data_loader):
    #     try:
    #         reference = latex_producer._idx2formulas(tgt4cal_loss)
    #         results = latex_producer(imgs)
    #     except RuntimeError:
    #         break

    print(result)

    context={'filePathName':filePathName,'result':result}
    return render(request,'index.html',context) 



# def predictImage(request):
# 	fileObj = request.FILES['filePath']
# 	fs = FileSystemStorage()
# 	filePathName = fs.save(fileObj.name, fileObj)
# 	filePathName = fs.url(filePathName)


# 	# model.eval()

# 	context = {'filePathName':filePathName}
# 	return render(request, 'index.html', context)

def handle_uploaded_file(f):
    name = str(datetime.datetime.now().strftime('%H%M%S')) + str(random.randint(0, 1000)) + str(f)
    path = default_storage.save(MEDIA_ROOT + '/' + name,
                                ContentFile(f.read()))
    return os.path.join(MEDIA_ROOT, path), name

# def predictImage(request):
#     if request.POST:
#         # imgtovec = Img2Vec()
#         file_path, file_name = handle_uploaded_file(request.FILES['filePath'])
#         # file2_path, file2_name = handle_uploaded_file(request.FILES['file2'])
#         # pic_one_vector = imgtovec.get_vec(Image.open(file1_path))
#         # pic_two_vector = imgtovec.get_vec(Image.open(file2_path))
#         # Using PyTorch Cosine Similarity
#         # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
#         # cos_sim = cos(torch.tensor(pic_one_vector).unsqueeze(0), torch.tensor(pic_two_vector).unsqueeze(0))

#         # print('\nCosine similarity: {0:.2f}\n'.format(float(cos_sim)))
#         context = {
#         	'filePathName':filePathName,
#         	'formula':predictedFormula,
#         	'post':True,
#         }
#         return render(request, "index.html", context)
    
#     else:    
#     	return render(request, "index.html", {'post': False})