from data.AgricultureVision.pre_process import *
from data.AgricultureVision.loader import * 
from torch.utils import data
import torch
import time
from tqdm import tqdm
import torchvision.transforms as st
import network
import utils
import torch.nn as nn
import os
import cv2







def main():
    
    output_dir = os.path.join(os.getcwd(), 'test_preds')
    bands = ['NIR', 'RGB']
    loader = AlgricultureDataset
    labels = land_classes
    nb_classes = len(land_classes)
    dataset = 'Agriculture'
    k_folder = 0
    k = 0
    input_size = [512, 512]
    scale_rate = 1.0/1.0
    seeds = 12345
    test_samples = 3729
    test_dict = get_real_test_list(root_folder = TEST_ROOT, data_folder=Data_Folder, name='Agriculture', bands=bands)
    test_files = (loadtestimg(test_dict))
    idlist=(loadids(test_dict))
    # all_gts = []
    num_class = len(labels)
    stride=600
    batch_size=4
    norm=False
    window_size=(512, 512)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    model = network.deeplabv3plus_resnet50(num_classes=7, output_stride=32)
    #utils.set_bn_momentum(model.backbone, momentum=0.01)
    checkpoint = torch.load('checkpoints/best_deeplabv3plus_resnet50_agr_os32.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    print("Model Loaded")
    #model = nn.DataParallel(model)
    model.to(device)
    del checkpoint  # free memory
    total_ids = 0
    for k in test_dict[IDS].keys():
        total_ids += len(test_dict[IDS][k])
    batcher = 0
    batch = []
    id_list = []
    for img, id in tqdm(zip(test_files, idlist), total=total_ids, leave=False):
        if batcher == total_ids: # Last image does not fit into batch of 4
            img = np.asarray(img, dtype='float32')
            img = st.ToTensor()(img)
            img = img / 255.0
            if norm:
                img = st.Normalize(*mean_std)(img)
            img = img.to(device, dtype=torch.float32)
            img = torch.unsqueeze(img, 0)  
            batch = [img, img, img, img]
            batched_image = torch.cat(batch)
            outputs = model(batched_image)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            current_pred = preds[0, :, :]
            filename = '{}.png'.format(id)
            cv2.imwrite(os.path.join(output_dir, filename), current_pred)
            
        else:
            img = np.asarray(img, dtype='float32')
            img = st.ToTensor()(img)
            img = img / 255.0
            if norm:
                img = st.Normalize(*mean_std)(img)
            img = img.to(device, dtype=torch.float32)
            img = torch.unsqueeze(img, 0)        
            batcher += 1
            batch.append(img)
            id_list.append(id)
            if batcher % batch_size == 0:
                batched_image = torch.cat(batch)
                batch = []
                outputs = model(batched_image)
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                for i in range(batch_size):
                    current_pred = preds[i, :, :]
                    current_id = id_list[i]
                    filename = '{}.png'.format(current_id)
                    cv2.imwrite(os.path.join(output_dir, filename), current_pred)
                id_list = []

        #stime = time.time()

def loadtestimg(test_files):

    id_dict = test_files[IDS]
    image_files = test_files[IMG]
    # mask_files = test_files[GT]

    for key in id_dict.keys():
        for id in id_dict[key]:
            if len(image_files) > 1:
                imgs = []
                for i in range(len(image_files)):
                    filename = image_files[i].format(id)
                    path, _ = os.path.split(filename)
                    if path[-3:] == 'nir':
                        # img = imload(filename, gray=True)
                        img = np.asarray(Image.open(filename), dtype='uint8')
                        img = np.expand_dims(img, 2)

                        imgs.append(img)
                    else:
                        img = imload(filename)
                        imgs.append(img)
                image = np.concatenate(imgs, 2)
            else:
                filename = image_files[0].format(id)
                path, _ = os.path.split(filename)
                if path[-3:] == 'nir':
                    # image = imload(filename, gray=True)
                    image = np.asarray(Image.open(filename), dtype='uint8')
                    image = np.expand_dims(image, 2)
                else:
                    image = imload(filename)
            # label = np.asarray(Image.open(mask_files.format(id)), dtype='uint8')

            yield image


def loadids(test_files):
    id_dict = test_files[IDS]

    for key in id_dict.keys():
        for id in id_dict[key]:
            yield id

        
if __name__ == '__main__':
    main()