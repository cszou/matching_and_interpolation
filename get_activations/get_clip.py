import clip
from tqdm import tqdm
from utils import *

def get_clip_encodings_from_index_vector(indices, dataloader, model, clip_device):
    # print(f'shape of indices {indices.shape}')
    top_images = get_images_from_indices(indices, dataloader.dataset).squeeze() #Need to squeeze for the model
    # print(top_images.shape)
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
    # top_kth_embedding = get_clip_encodings_from_index_vector(indices, dataloader, model, clip_device)
    # embeddings_tensor = top_kth_embedding
    print(f'overall embeddings shape: {embeddings_tensor.shape}')

    return embeddings_tensor


if __name__ == '__main__':
    indices_m1 = torch.load('m1.result.pth.tar')['top_dataset_indices']
    indices_m2 = torch.load('m2.result.pth.tar')['top_dataset_indices']
    # print(indices_m1)
    embeddings_m1 = {}
    embeddings_m2 = {}
    for k in indices_m1.keys():
        # print(k, v.shape)
        embeddings_m1[k] = get_clip_encodings_from_index_tensor(indices_m1[k])
        embeddings_m2[k] = get_clip_encodings_from_index_tensor(indices_m2[k])