# generate_embedding.py
from facenet_pytorch import MTCNN, InceptionResnetV1
import os, json
from PIL import Image
from tqdm import tqdm

inp = input('Please input the relative path of the dataset folder: ')
dataset_path = os.path.join(os.getcwd(), inp)
embed_filename = os.path.join(dataset_path, "embeddings.json")

embeddings = dict()
if os.path.exists(embed_filename):
    embeddings = json.load(open(embed_filename, "r"))

file_list = os.listdir(dataset_path)

mtcnn = MTCNN(image_size=128, device='cuda')
resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()

for fname in tqdm(file_list):
    try:
        if fname[-4:] != '.jpg' or fname in embeddings:
            continue
        img = Image.open(os.path.join(dataset_path, fname))
        img_cropped = mtcnn(img, save_path=os.path.join(dataset_path, 'aligned', fname))
        embedding = resnet(img_cropped.unsqueeze(0).cuda())
        embeddings[fname] = embedding.cpu().detach().numpy().tolist()[0]
    except:
        print(f'There\'s something wrong with this img: {fname}')
    finally:
        pass

json.dump(embeddings, open(embed_filename, "w"))