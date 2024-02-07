# from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
# from reportlab.lib.pagesizes import A4, letter
# from PIL import Image as PImage
# from reportlab.lib.styles import ParagraphStyle
import torch.nn.functional as F
import torch
from utils import get_clip_encoding_dataloader, get_images_from_indices
import clip
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

#TODO look at the KT?
def get_clip_encodings_from_index_vector(indices,  dataloader, model, clip_device):
    #print(f'indices for clip image have shape: {indices.shape}')

    top_images = get_images_from_indices(indices, 1, dataloader).squeeze() #Need to squeeze for the model

    with torch.no_grad():
        image_features = model.encode_image(top_images.to(clip_device))

    #print('encoded images shape: ', image_features.shape)
    return image_features




def get_clip_encodings_from_index_tensor(indices, topk=10,  batch_size = 256, num_workers=10):
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', clip_device)
    model = model.eval()
    dataloader = get_clip_encoding_dataloader(preprocess, batch_size, num_workers)
    print('grabbing embeddings of top images for all channels')
    all_embeddings = []
    for index_of_top_kth_images in tqdm(range(topk)):
        top_kth_embedding = get_clip_encodings_from_index_vector(indices[index_of_top_kth_images].unsqueeze(0), dataloader, model, clip_device)
        all_embeddings.append(top_kth_embedding.unsqueeze(0))

    embeddings_tensor = torch.vstack(all_embeddings)
    print(f'overall embeddings shape: {embeddings_tensor.shape}')

    return embeddings_tensor

#x1 and x2 should be embeddings for a channel of shape [topk, embed_dim]
def get_mean_similarity(x1, x2):
    #print(x1.shape,x2.shape)
    sim = 0
    count = 0
    for embed1 in x1:
      for embed2 in x2:
        #if count ==0:
        #  print(embed1.shape,embed2.shape)
        sim = sim + F.cosine_similarity(embed1, embed2, dim=0)
        count = count+1
    sim = sim/count
    return sim

#x1 and x2 should be embeddings for a channel of shape [topk, embed_dim]
def get_similarity_of_mean(x1, x2):
    #print(x1.shape, x2.shape)

    x1 = x1.mean(dim=0)
    x2 = x2.mean(dim=0)
    #print(f'x1 shape: {x1.shape}')
    #print('x1 norm', torch.linalg.vector_norm(x1))
    #print('x2 norm', torch.linalg.vector_norm(x2))
    sim = F.cosine_similarity(x1, x2, dim=0)
    #print(f'sim {sim}')

    #exit(0)
    return sim



def gen_cosine_sim_tensor(embeddings1, embeddings2):
    print('embeddings passed to cos sim: ', embeddings1.shape)
    embeddings1 = embeddings1.permute(1, 0, 2)  ## [channels, topk, embedding]
    embeddings2 = embeddings2.permute(1, 0, 2)
    print('embeddings shape: ', embeddings1.shape)

    #normalize the embeddings for easier similarity calculations
    embeddings1 = F.normalize(embeddings1, dim=-1)
    embeddings2 = F.normalize(embeddings2, dim=-1)
    embeddings1 = embeddings1.unsqueeze(1)
    embeddings2 = embeddings2.unsqueeze(0).permute(0,1,3,2)
    print('embeddings1 shape: ', embeddings1.shape)
    print('embeddings2 shape: ', embeddings2.shape)
    similarities =  torch.matmul(embeddings1, embeddings2)
    #Generate a [channels, channels,topk,topk] tensor with the cosine similarity 
    print(f'similarites shape: {similarities.shape}')

    return similarities

def gen_cosine_sim_tensor_full_loop(embeddings1, embeddings2):
    print('embeddings passed to cos sim: ', embeddings1.shape)
    embeddings1 = embeddings1.permute(1, 0, 2)  ## [channels, topk, embedding]
    embeddings2 = embeddings2.permute(1, 0, 2)
    print('embeddings shape: ', embeddings1.shape)

    cos = torch.nn.CosineSimilarity()
    results_looped = torch.zeros(256,256,10,10)
    print(embeddings1.shape, embeddings2.shape)
    for i in range(256):
        if i%50==0:
            print(f'{i} of {results_looped.shape[0]}')
        for j in range(256):
            temp1 = embeddings1[i]
            temp2 = embeddings2[j]
            results_looped[i,j] = cos(temp1[...,None],temp2.t()[None,...])
    #Generate a [channels, channels,topk,topk] tensor with the cosine similarity 
    print(f'similarites shape: {results_looped.shape}')

    return results_looped

### FULL 256x256x10x10 vs 256x256
### First means using Torch's normalize which gives incorrect norms
### Second is less flexible but can use the numpy cosine that works better.
def get_overall_cos_sim_results(init_indices, final_indices):

    #Need to unsqueeze for my get fn
    init_embeddings = get_clip_encodings_from_index_tensor(init_indices)
    final_embeddings = get_clip_encodings_from_index_tensor(final_indices)
    print('embeddings shape')
    print(init_embeddings.shape)
    init_similarities = gen_cosine_sim_tensor(init_embeddings, init_embeddings)
    final_similarites = gen_cosine_sim_tensor(init_embeddings, final_embeddings)
    return init_similarities, final_similarites

def get_embeddings(init_indices, final_indices):

    #Need to unsqueeze for my get fn
    init_embeddings = get_clip_encodings_from_index_tensor(init_indices)
    final_embeddings = get_clip_encodings_from_index_tensor(final_indices)

    return init_embeddings, final_embeddings

def get_overall_cos_sim_results_full_loop(init_indices, final_indices):

     #Need to unsqueeze for my get fn
    init_embeddings = get_clip_encodings_from_index_tensor(init_indices)
    final_embeddings = get_clip_encodings_from_index_tensor(final_indices)
    print('embeddings shape')
    print(init_embeddings.shape)
    init_similarities = gen_cosine_sim_tensor_full_loop(init_embeddings, init_embeddings)
    final_similarites = gen_cosine_sim_tensor_full_loop(final_embeddings, final_embeddings)
    return init_similarities, final_similarites

    
def get_overall_cos_sim_results_with_mean_embeddings(init_indices, final_indices):
     #Need to unsqueeze for my get fn
    init_embeddings = get_clip_encodings_from_index_tensor(init_indices)
    final_embeddings = get_clip_encodings_from_index_tensor(final_indices)
    print('embeddings shape')
    print(init_embeddings.shape)
    init_embeddings = init_embeddings.mean(dim=0)
    final_embeddings = final_embeddings.mean(dim=0)
    print(f'mean of embeddings shape {init_embeddings.shape}')
    init_embeddings = F.normalize(init_embeddings,dim=1)
    final_embeddings = F.normalize(final_embeddings,dim=1)

    init_similarities = init_embeddings @ init_embeddings.T
    final_similarites = init_embeddings @ final_embeddings.T
    print(f'similarities shape: {init_similarities.shape}')
    return init_similarities, final_similarites

if __name__ == "__main__":

    version_number = '0.1'
    print("Version Number: " + version_number)


    indices_m1 = torch.load('m1.result.pth.tar')['top_dataset_indices']
    indices_m2 = torch.load('m2.result.pth.tar')['top_dataset_indices']
    # print(indices_m1)
    embeddings_m1 = {}
    embeddings_m2 = {}
    similarities = {}
    for k in indices_m1.keys():
        print(k)
        embeddings_m1[k] = get_clip_encodings_from_index_tensor(indices_m1[k])
        embeddings_m2[k] = get_clip_encodings_from_index_tensor(indices_m2[k])
        similarities[k] = gen_cosine_sim_tensor(embeddings_m1[k], embeddings_m2[k])
        print('shape:', similarities[k].shape)
        print("diag:", similarities[k].diag())
        print('mean:', similarities[k].mean())
        print('samples:', similarities[k][0, :10])
        print('max by row: ', similarities[k].max(dim=1)[0])
        print('min by row: ', similarities[k].min(dim=1)[0])
        print('max - self', similarities[k].max(dim=1)[0]-similarities[k].diag())
    print('Done testing!')

