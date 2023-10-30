# Import Libraries
import os
import torch, cv2
import open_clip
from PIL import Image
from utils.config import *
from utils.common_utils import read_json_data
from utils.dataset import CaptionInContext
from utils.text_utils import get_text_metadata
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim

from torch.utils.data import DataLoader
from utils.dataset_utils import PadCollate2
import torch.nn.functional as F
import matplotlib.pyplot as pltc

import torchvision
import numpy as np
from tqdm import tqdm
import argparse

import os.path
import torch
import itertools as it
from torch.utils.data import Dataset
from utils.dataset_utils import modify_caption_replace_entities
from utils.common_utils import read_json_data
from utils.config import num_boxes, DATA_DIR
from utils.custom_transforms.data_aug import *


transform3 = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
transform2 = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.ColorJitter(hue=.2, saturation=.2), transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
transform1 = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])


def load_images(imgPath, bboxes):
    images = []
    image = Image.open(imgPath)
    for j, bbox in enumerate(bboxes):
        img1 = image.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
        if img1.size[0] > 0 and img1.size[1] > 0:
            images.append(img1)
    return images



class AugDataset(Dataset):
    """Custom dataset class for Out-of-Context Detection"""                                         ################################

    def __init__(self, metadata_file, mode, transforms, text_field=None, max_samples=None, slice_start=None, slice_end=None):
        """
            Initializes the dataset object
            Args:
                metadata_file (string): Path to the json file with annotations.
                mode (string): train, val, test.
                transforms (callable): Transform to be applied on a sample.
                text_field (torchtext.Field): Vocab object for applying text transformations (used for Glove and Fasttext embeddings only)
            Returns:
                None
        """
        self.data = read_json_data(metadata_file)
        self.mode = mode
        self.transforms = transforms
        self.text_field = text_field
        self.max_samples = max_samples

        self.flip_rotate_transform = Sequence(
            [RandomHorizontalFlip(0.8), RandomScale(0.2, diff=True), RandomRotate(10)])

        if max_samples is not None:
            self.data = self.data[:max_samples]
        #########################################
        if slice_start is not None or slice_end is not None:
        ###########################################
            self.data = self.data[slice_start:slice_end]    
    
    
    def __getitem__(self, index):
        """
            Returns sample corresponding to the index `index`
        """
        img_data = self.data[index]
        img_path = os.path.join(DATA_DIR, img_data['img_local_path'])
        bboxes = img_data['maskrcnn_bboxes'][:10]

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        try:
            img_aug, bboxes_aug = self.flip_rotate_transform(img, np.array(bboxes))
            bboxes_aug = bboxes_aug.tolist()
            bboxes = list(it.islice(it.cycle(bboxes_aug), num_boxes - 1))
            img = img_aug
        except:
            pass

        if self.mode == 'test':
            idx1 = random.randint(0, 1)
            cap_key1 = 'caption1' if idx1 == 0 else 'caption2'
            caption1 = img_data[cap_key1]
            caption1 = modify_caption_replace_entities(caption1)

            while True:
                idx2 = random.randint(0, 1)
                cap_key2 = 'caption1' if idx2 == 0 else 'caption2'
                tgt_index = random.randint(0, len(self.data) - 1)
                caption2 = self.data[tgt_index][cap_key2]
                caption2 = modify_caption_replace_entities(caption2)
                if caption1 != caption2: #Aisa mazak nhi krne ka
                    break
        else:
            src_captions = img_data['articles']
            caption1 = src_captions[random.randint(0, len(src_captions) - 1)]['caption_modified']

            while True:
                tgt_index = random.randint(0, len(self.data) - 1)
                tgt_captions = self.data[tgt_index]['articles']
                caption2 = tgt_captions[random.randint(0, len(tgt_captions) - 1)]['caption_modified']
                if caption1 != caption2:
                    break
        text1 = caption1
        text2 = caption2
        
        return text1, bboxes, img_path,text2
        # return img_path, img_aug1, img_aug2, images[0], images[1], images[2], images[3], images[4], images[5], images[6], images[7], images[8], images[9], bboxes, text

    def __len__(self):
        """
            Returns length of the dataset
        """
        return len(self.data)
    

# Word Embeddings
text_field, word_embeddings, vocab_size = get_text_metadata()
train_dataset = AugDataset(metadata_file=os.path.join(DATA_DIR, 'annotations', 'train_data.json'),
                                 transforms=transform3, mode='train', text_field=text_field, max_samples=16000)

val_dataset  = AugDataset(metadata_file=os.path.join(DATA_DIR, 'annotations', 'val_data.json'),
                               transforms=transform3, mode='val', text_field=text_field, max_samples=3200)

# Creating data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=6, shuffle=True, collate_fn=PadCollate2())

val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=6, shuffle=False, collate_fn=PadCollate2())


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device, jit=False)
tokenizer = open_clip.get_tokenizer('ViT-B-32')


class CLIPModel(nn.Module):
    def __init__(self, model):
        super(CLIPModel, self).__init__()

        # Preprocessing layers
        self.model = model
        self.fc = nn.Linear(512, 512)
        self.fc1 = nn.Linear(5120, 2560)
        self.fc2 = nn.Linear(2560, 512)

        # Set requires_grad to True for the parameters
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.transformer.resblocks[-1].mlp.parameters():
            param.requires_grad_(True)

        for param in self.model.visual.transformer.resblocks[-1].mlp.parameters():
            param.requires_grad_(True)

        for param in self.model.ln_final.parameters():
           param.requires_grad = True

        for param in self.fc.parameters():
            param.requires_grad = True

        for param in self.fc1.parameters():
            param.requires_grad = True

        for param in self.fc2.parameters():
            param.requires_grad = True

    def forward(self, orgImg, img_aug1, img_aug2, images, tokenisedCaption,tokenisedCaption1):

        # Create a list to store the encoded features for each image in the batch
        feature_list = []
        bbox_list = []
        bbox_list1 = []
        feature_list_img = []
        feature_list_aug_text = []
        encoded_orgImg = self.fc(self.model.encode_image(orgImg.to(device)))
        encoded_orgImg = encoded_orgImg / encoded_orgImg.norm(dim=-1, keepdim=True)
        feature_list.append(encoded_orgImg)
        feature_list_img.append(encoded_orgImg)

        encoded_img_aug1 = self.fc(self.model.encode_image(img_aug1.to(device)))
        encoded_img_aug1 = encoded_img_aug1 / encoded_img_aug1.norm(dim=-1, keepdim=True)
        feature_list.append(encoded_img_aug1)
        feature_list_img.append(encoded_img_aug1)

        encoded_img_aug2 = self.fc(self.model.encode_image(img_aug2.to(device)))
        encoded_img_aug2 = encoded_img_aug2 / encoded_img_aug2.norm(dim=-1, keepdim=True)
        feature_list.append(encoded_img_aug2)
        feature_list_img.append(encoded_img_aug2)

        for image in images:
            encoded_image = self.fc(self.model.encode_image(image.to(device)))
            encoded_image_norm = encoded_image / encoded_image.norm(dim=-1, keepdim=True)
            bbox_list.append(encoded_image_norm)
     
            bbox_list1.append(encoded_image_norm)
        
        bbox_list1.append(encoded_orgImg)

            
        
        bbox_list = self.fc1(bbox_list)
        bbox_list = bbox_list / bbox_list.norm(dim=-1, keepdim=True)

        bbox_list = self.fc2(bbox_list)
        bbox_list = bbox_list / bbox_list.norm(dim=-1, keepdim=True)

        feature_list.append(bbox_list)
        feature_list_img.append(bbox_list)

        encoded_text = self.fc(self.model.encode_text(tokenisedCaption.to(device)))
        encoded_text = encoded_text / encoded_text.norm(dim=-1, keepdim=True)

        encoded_text1 = self.fc(self.model.encode_text(tokenisedCaption1.to(device)))
        encoded_text1 = encoded_text1 / encoded_text1.norm(dim=-1, keepdim=True)

        feature_list_aug_text.append(encoded_text1)
        feature_list_aug_text.append(encoded_text1)
        feature_list_aug_text.append(encoded_text1)
        feature_list_aug_text.append(encoded_text1)

        # feature_list_cap = feature_list[:]
        feature_list.append(encoded_text)
        bbox_list1.append(encoded_text)
        bbox_list1.append(encoded_text1)
        
        # feature_list_cap.insert(0,encoded_text)
        return feature_list, bbox_list1,feature_list_img,feature_list_aug_text

clipModel = CLIPModel(model)
model_name = "ClipModelResUnfreezeSCL_temp_1_loss3_16k_val_change_weighted_04_CL_1_5_mmlosstrain_scaling"

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        # print(self.temperature)s
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        # print(self.base_temperature)

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda:1')
        #           if features.is_cuda
        #           else torch.device('cuda:0'))
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
            
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
            
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # print(logits_max)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # print(mean_log_prob_pos)
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    

optimizer = optim.Adam([p for p in clipModel.parameters() if p.requires_grad],lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)

print("Total Params", sum(p.numel() for p in clipModel.parameters() if p.requires_grad))
print("Img Model", sum(p.numel() for p in clipModel.model.parameters() if p.requires_grad))


import math
temp = 0.01
criterion = SupConLoss(temperature=temp)

def train_model(epoch):

    train_loss = 0.
    clipModel.train()
    clipModel.to(device)
    optimizer.zero_grad()
    clipModel.zero_grad()

    # Training loop
    for batch_idx, (caption1, bboxes, img_path,caption2) in enumerate(tqdm(train_loader)):
        batch = len(bboxes)
        final_tensor = torch.zeros([batch_size,2])
        encoded_images_lists = [[] for _ in range(5)]
        # encoded_images_lists_img_aug = [[] for _ in range(4)]
        # encoded_images_lists_text_aug = [[] for _ in range(4)]

        # encoded_images_lists_cap = [[] for _ in range(14)]
        with torch.set_grad_enabled(True):      
            for i, imgPath in enumerate(img_path):
                images = load_images(imgPath, bboxes[i])
                if len(images) == 10:
                    tokenisedCaption1 = tokenizer([caption1[i][:77]])
                    tokenisedCaption2 = tokenizer([caption2[i][:77]])
                    
                    imgPath = os.path.join(DATA_DIR, imgPath)
                    img = Image.open(imgPath)
                    img_aug1 = transform1(preprocess(img)).unsqueeze(0)
                    img_aug2 = transform2(preprocess(img)).unsqueeze(0)
                    orgImg = transform3(preprocess(img)).unsqueeze(0)
                    images = [transform3(preprocess(imgs)).unsqueeze(0) for imgs in images]

                    features, _ , _ , _= clipModel(orgImg, img_aug1, img_aug2, images, tokenisedCaption1,tokenisedCaption2)
                    for i, feature in enumerate(features):
                        encoded_images_lists[i].append(feature)
                    
                    # _, _,features_img_aug, _ = clipModel(orgImg, img_aug1, img_aug2, images, tokenisedCaption1,tokenisedCaption2)
                    # for i, feature in enumerate(features_img_aug):
                    #     encoded_images_lists_img_aug[i].append(feature)
                    
                    # _, _,_, features_text_aug = clipModel(orgImg, img_aug1, img_aug2, images, tokenisedCaption1,tokenisedCaption2)
                    # for i, feature in enumerate(features_text_aug):
                    #     encoded_images_lists_text_aug[i].append(feature)
                    
                    max1,max2 = myFunc(orgImg, img_aug1, img_aug2, images, tokenisedCaption1, tokenisedCaption2 )
                    final_tensor[i,0] = max1
                    final_tensor[i,1] = max2


                    # for j, feature in enumerate(features_cap):
                        # encoded_images_lists_cap[j].append(feature)  
                else:
                    pass
             
            encoded_images_lists = [torch.stack(encoded_images_list) for encoded_images_list in encoded_images_lists]

            # encoded_images_lists_img_aug = [torch.stack(encoded_images_list) for encoded_images_list in encoded_images_lists_img_aug]

            # encoded_images_lists_text_aug = [torch.stack(encoded_images_list) for encoded_images_list in encoded_images_lists_text_aug]

            # encoded_images_lists_cap = [torch.stack(encoded_images_list) for encoded_images_list in encoded_images_lists_cap]
         
            encoded_images_lists = torch.stack(encoded_images_lists, dim=1)

            # encoded_images_lists_img_aug = torch.stack(encoded_images_lists_img_aug, dim=1)

            # encoded_images_lists_text_aug = torch.stack(encoded_images_lists_text_aug, dim=1)

            # img_text_aug_feat = torch.cat([encoded_images_lists_img_aug,encoded_images_lists_text_aug], dim=0)

            # encoded_images_lists_cap = torch.stack(encoded_images_lists_cap, dim=1)

            final_tensor = final_tensor.to(device)
            loss = math.exp(-epochs/4)*criterion(encoded_images_lists) + margin_rank_loss(final_tensor[:, 0], final_tensor[:, 1], torch.ones(final_tensor.shape[0]).to(device)) #+ 1.5*criterion(img_text_aug_feat)
            train_loss += float(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # clipModel.to(device)

            torch.cuda.empty_cache()
            torch.cuda.synchronize() 
            del encoded_images_lists

            print('For Batch: {}, Total Loss: {:.4f}, Current Loss: {:.4f}'.format(int(batch_idx), train_loss / len(train_loader), loss))

    print(' Train Epoch: {} Loss: {:.4f}'.format(epoch, train_loss / len(train_loader)))




def myFunc(orgImg,img_aug1,img_aug2,images,textClip1,textClip2):

    _, box_list1,_,_ = clipModel(orgImg, img_aug1, img_aug2, images, textClip1, textClip2)

    imgFeatures = box_list1[:-3]
    textFeatures1 = box_list1[-2]
    textFeatures2 = box_list1[-1]

    # Convert the list of tensors into a single tensor for efficient computation
    imgFeatures_tensor = torch.cat(imgFeatures, dim=0)

    # Compute the dot product of textFeatures1 and imgFeatures
    dot_product1 = torch.mm(textFeatures1, imgFeatures_tensor.t())

    # Compute the dot product of textFeatures2 and imgFeatures
    dot_product2 = torch.mm(textFeatures2, imgFeatures_tensor.t())

    # Convert the results back to lists
    dot_product1_list = dot_product1.squeeze().tolist()
    dot_product2_list = dot_product2.squeeze().tolist()

    return max(dot_product1_list), max(dot_product2_list)
    


def evaluate_model(epoch):
    val_loss = 0.
    correct = 0.
    total = 0.
    clipModel.eval()
    clipModel.to(device)

    # Validation  loop
    for batch_idx, (caption1, bboxes, img_path,caption2) in enumerate(tqdm(val_loader)):
        batch = len(bboxes)
        final_tensor = torch.zeros([batch_size,2])
        # encoded_images_lists_cap = [[] for _ in range(14)]
        with torch.no_grad():      
            for i, imgPath in enumerate(img_path):
                images = load_images(imgPath, bboxes[i])
                if len(images) == 10:
                    tokenisedCaption1 = tokenizer([caption1[i][:77]])
                    tokenisedCaption2 = tokenizer([caption2[i][:77]])
                    
                    imgPath = os.path.join(DATA_DIR, imgPath)
                    img = Image.open(imgPath)

                    img_aug1 = transform1(preprocess(img)).unsqueeze(0)
                    img_aug2 = transform2(preprocess(img)).unsqueeze(0)
                    orgImg = transform3(preprocess(img)).unsqueeze(0)
                    
                    images = [transform3(preprocess(imgs)).unsqueeze(0) for imgs in images]

                    max1,max2 = myFunc(orgImg, img_aug1, img_aug2, images, tokenisedCaption1, tokenisedCaption2 )
                    final_tensor[i,0] = max1
                    final_tensor[i,1] = max2

                    # _, box_list1,_,_ = clipModel(orgImg, img_aug1, img_aug2, images, tokenisedCaption1, tokenisedCaption2)

                    # imgFeatures = box_list1[:-3]
                    # textFeatures1 = box_list1[-2]
                    # textFeatures2 = box_list1[-1]


                    # for j, feature in enumerate(features_cap):
                    #     encoded_images_lists_cap[j].append(feature)  
                else:
                    pass
            final_tensor = final_tensor.to(device)
            loss = margin_rank_loss(final_tensor[:, 0], final_tensor[:, 1], torch.ones(final_tensor.shape[0]).to(device))

            val_loss += float(loss.item())
            correct += torch.sum(final_tensor[:, 0] > final_tensor[:, 1]).item()
            total += batch 
            # clipModel.to(device)

            torch.cuda.empty_cache()
            torch.cuda.synchronize() 

            print('For Batch: {}, Total Loss: {:.4f}, Current Loss: {:.4f}, Acc: {:.2f}'.format(int(batch_idx), val_loss / len(val_loader), loss, correct / total))

    print(' Validation Epoch: {} Loss: {:.4f} Acc: {:.2f} '.format(epoch, val_loss / len(val_loader), correct / total))
    return val_loss




def train_joint_model():
    """
        Performs training and validation on the dataset
    """
    try:
        print("Loading Saved Model")
        checkpoint = torch.load(BASE_DIR + 'models/' + model_name + '.pt')
        clipModel.load_state_dict(checkpoint)
        print("Saved Model successfully loaded")
        clipModel.eval()
        best_loss = eval_validation_loss()
    except:
        best_loss = np.Inf
    early_stop = False
    counter = 0
    for epoch in range(1, epochs + 1):
        # Training epoch
        print("Epoch Starts")
        train_model(epoch)
        # Validation epoch
        avg_test_loss = evaluate_model(epoch)
        scheduler.step(avg_test_loss)
        if avg_test_loss <= best_loss:
            counter = 0
            best_loss = avg_test_loss
            torch.save(clipModel.state_dict(), 'models/' + model_name + '.pt')
            print("Best model saved/updated..")
            torch.cuda.empty_cache()
        else:
            counter += 1
            if counter >= patience:
                early_stop = True
        # If early stopping flag is true, then stop the training
        if early_stop:
            print("Early stopping")
            break


def eval_validation_loss():
    """
        Computes validation loss on the saved model, useful to resume training for an already saved model
    """
    val_loss = 0.
    with torch.no_grad():
        for batch_idx, (caption, bboxes, img_path) in enumerate(tqdm(val_loader, desc='')):

            encoded_images_lists = [[] for _ in range(14)]
          
            for i, imgPath in enumerate(img_path):
                tokenisedCaption = tokenizer([caption[i][:77]])
                
                imgPath = os.path.join(DATA_DIR, imgPath)
                img = Image.open(imgPath)

                img_aug1 = transform1(preprocess(img)).unsqueeze(0)
                img_aug2 = transform2(preprocess(img)).unsqueeze(0)
                orgImg = transform3(preprocess(img)).unsqueeze(0)
                images = load_images(imgPath, bboxes[i])
                images = [transform3(preprocess(imgs)).unsqueeze(0) for imgs in images]

                features = clipModel(orgImg, img_aug1, img_aug2, images, tokenisedCaption)
                for i, feature in enumerate(features):
                    encoded_images_lists[i].append(feature)
            
            encoded_images_lists = [torch.stack(encoded_images_list) for encoded_images_list in encoded_images_lists]
            
            encoded_images_lists = torch.stack(encoded_images_lists, dim=1)
            loss = criterion(encoded_images_lists)
            val_loss += float(loss.item())
            # clipModel.to(device)

            torch.cuda.empty_cache()
            torch.cuda.synchronize() 
            del encoded_images_lists

            print('For Batch: {}, Total Loss: {:.4f}, Current Loss: {:.4f}'.format(int(batch_idx), val_loss / len(train_loader), loss))

    print(' Validation Epoch: {} Loss: {:.4f}'.format( val_loss / len(train_loader)))
    return val_loss



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--mode', type=str, default='test',
                        help="mode, {'" + "train" + "', '" + "eval" + "'}")
    args = parser.parse_args()
    if args.mode == 'train':
        train_joint_model()
    # elif args.mode == 'eval':
    #     test_match_accuracy()
