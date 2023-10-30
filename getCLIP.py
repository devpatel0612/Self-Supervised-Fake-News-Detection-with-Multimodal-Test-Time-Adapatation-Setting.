import os
os.environ["TFHUB_CACHE_DIR"] = '/home/manogna/Cosmos/COSMOS/tfhub_cache'
import torch, cv2
from utils.config import *
import clip
from PIL import Image
from utils.common_utils import read_json_data
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

test_samples = read_json_data(os.path.join(DATA_DIR, 'annotations', 'test_data.json'))
ours_correct = 0

for i, v_data in enumerate(test_samples):
    img_path = os.path.join(DATA_DIR, v_data["img_local_path"])

    img = Image.open(img_path)

    bounding_boxes = v_data['maskrcnn_bboxes']

    # Iterate through the list of bounding boxes
    min_diff = float('inf')  # Initialize the minimum difference as infinity
    boxMax_index = None  # Initialize the index of the bounding box with minimum difference as None

    for index, bbox in enumerate(bounding_boxes):
        img1 = img.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))

        if img1.size[0] > 0 and img1.size[1] > 0:
            image = preprocess(img1).unsqueeze(0).to(device)
            
            max_length = 77  # Replace this with the appropriate maximum length for the CLIP model
            truncated_caption1 = v_data['caption1'][:max_length]
            truncated_caption2 = v_data['caption2'][:max_length]
            
            text = clip.tokenize([truncated_caption1,truncated_caption2]).to(device)


            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                
                logits_per_image, logits_per_text = model(image, text)
                # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                logit = logits_per_image.cpu().numpy()

                diff = abs(logit[0][0] - logit[0][1])
                
                # Update the minimum difference and its index if a smaller difference is found
                if diff < min_diff:
                    min_diff = diff
                    boxMax_index = index

    # Get the bounding box with the minimum difference
    boxMax = bounding_boxes[boxMax_index]

    # Iterate through each caption and apply the CLIP model
    max_length = 77  # Replace this with the appropriate maximum length for the CLIP model
    truncated_caption1 = v_data['caption1'][:max_length]
    truncated_caption2 = v_data['caption2'][:max_length]

    text = torch.cat([clip.tokenize(truncated_caption1), clip.tokenize(truncated_caption2)]).to(device)
        # text2 = clip.tokenize([caption]).to(device)

    with torch.no_grad():
        # image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    cosine_similarity = torch.nn.functional.cosine_similarity(text_features[0], text_features[1], dim=-1)

    if min_diff < 0.3:
        if cosine_similarity > 0.65:
            pred_context = 0
        else:
            pred_context = 1
    else:
        pred_context = 0

    actual_context = int(v_data['context_label'])
    if pred_context == actual_context:
        ours_correct += 1

print("CLIP - COSMOS Modified Accuracy", ours_correct / len(test_samples))
