""" Evaluation script to evaluate Out-of-Context Detection Accuracy"""
import os
os.environ["TFHUB_CACHE_DIR"] = '/home/manogna/Cosmos/COSMOS/tfhub_cache'
import cv2
from utils.config import *
from utils.text_utils import get_text_metadata
from model_archs.models import CombinedModelMaskRCNN
from utils.common_utils import read_json_data
from utils.eval_utils import is_bbox_overlap, top_bbox_from_scores


# Word Embeddings
text_field, word_embeddings, vocab_size = get_text_metadata()

# Choose Model as per User Input
print("-------Please choose model here to load------")
print("Enter 1 for Baseline Cosmos Pretrained Model")
print("Enter 2 for MyModel trained on 35k training dataset")
print("Enter 3 for MyModel trained on 1L training dataset")
print("Enter 4 for MyModel trained on 1.6L training dataset")

modelNumber = int(input("Choose Model here: "))
print(modelNumber)
if modelNumber == 1:
    model_name = 'baselineModel'
    combined_model = CombinedModelMaskRCNN(hidden_size=300, use=True).to(device)
    print(model_name)
elif modelNumber == 2:
    model_name = 'MyModel35k'
    combined_model = CombinedModelMaskRCNN(hidden_size=300, use=True).to(device)
elif modelNumber == 3:
    model_name = 'MyModel1L'
    combined_model = CombinedModelMaskRCNN(hidden_size=300, use=True).to(device)
elif modelNumber == 4:
    model_name = 'test_80'
    combined_model = CombinedModelMaskRCNN(hidden_size=300, use=True).to(device)

else:
    print("Model Number Chosen incorrect")
    exit()


def get_scores(v_data):
    """
        Computes score for the two captions associated with the image

        Args:
            v_data (dict): A dictionary holding metadata about on one data sample

        Returns:
            score_c1 (float): Score for the first caption associated with the image
            score_c2 (float): Score for the second caption associated with the image
    """
    checkpoint = torch.load(BASE_DIR + 'models/' + model_name + '.pt')
    combined_model.load_state_dict(checkpoint)
    combined_model.to(device)
    combined_model.eval()

    img_path = os.path.join(DATA_DIR, v_data["img_local_path"])
    bbox_list = v_data['maskrcnn_bboxes']
    bbox_classes = [-1] * len(bbox_list)
    img = cv2.imread(img_path)
    img_shape = img.shape[:2]
    bbox_list.append([0, 0, img_shape[1], img_shape[0]])  # For entire image (global context)
    bbox_classes.append(-1)
    cap1 = v_data['caption1_modified']
    cap2 = v_data['caption2_modified']

    img_tensor = [torch.tensor(img).to(device)]
    bboxes = [torch.tensor(bbox_list).to(device)]
    bbox_classes = [torch.tensor(bbox_classes).to(device)]

    if embed_type != 'use':
        # For Glove, Fasttext embeddings
        cap1_p = text_field.preprocess(cap1)
        cap2_p = text_field.preprocess(cap2)
        embed_c1 = torch.stack([text_field.vocab.vectors[text_field.vocab.stoi[x]] for x in cap1_p]).unsqueeze(
            0).to(device)
        embed_c2 = torch.stack([text_field.vocab.vectors[text_field.vocab.stoi[x]] for x in cap2_p]).unsqueeze(
            0).to(device)
    else:
        # For USE embeddings
        embed_c1 = torch.tensor(use_embed([cap1]).numpy()).to(device)
        embed_c2 = torch.tensor(use_embed([cap2]).numpy()).to(device)

    with torch.no_grad():
        z_img, z_t_c1, z_t_c2 = combined_model(img_tensor, embed_c1, embed_c2, 1, [embed_c1.shape[1]],
                                               [embed_c2.shape[1]], bboxes, bbox_classes)

    z_img = z_img.permute(1, 0, 2)
    z_text_c1 = z_t_c1.unsqueeze(2)
    z_text_c2 = z_t_c2.unsqueeze(2)

    # Compute Scores
    score_c1 = torch.bmm(z_img, z_text_c1).squeeze()
    score_c2 = torch.bmm(z_img, z_text_c2).squeeze()
    
    return score_c1, score_c2


def evaluate_context_with_bbox_overlap(v_data):
    """
        Computes predicted out-of-context label for the given data point

        Args:
            v_data (dict): A dictionary holding metadata about on one data sample

        Returns:
            context_label (int): Returns 0 if its same/similar context and 1 if out-of-context
    """
    bboxes = v_data['maskrcnn_bboxes']
    score_c1, score_c2 = get_scores(v_data)
    textual_sim = float(v_data['bert_base_score'])

    top_bbox_c1 = top_bbox_from_scores(bboxes, score_c1)
    top_bbox_c2 = top_bbox_from_scores(bboxes, score_c2)
    bbox_overlap = is_bbox_overlap(top_bbox_c1, top_bbox_c2, iou_overlap_threshold)
    if bbox_overlap:
        # Check for captions with same context : Same grounding with high textual overlap (Not out of context)
        if textual_sim >= textual_sim_threshold:
            context = 0
        # Check for captions with different context : Same grounding with low textual overlap (Out of context)
        else:
            context = 1
        return context
    else:
        # Check for captions with same context : Different grounding (Not out of context)
        return 0


if __name__ == "__main__":
    """ Main function to compute out-of-context detection accuracy"""

    test_samples = read_json_data(os.path.join(DATA_DIR, 'annotations', 'test_data.json'))
    ours_correct = 0
    lang_correct = 0

    for i, v_data in enumerate(test_samples):
        actual_context = int(v_data['context_label'])
        language_context = 0 if float(v_data['bert_base_score']) >= textual_sim_threshold else 1
        pred_context = evaluate_context_with_bbox_overlap(v_data)

        if pred_context == actual_context:
            ours_correct += 1

        if language_context == actual_context:
            lang_correct += 1

    print("Cosmos Accuracy", ours_correct / len(test_samples))
    print("Language Baseline Accuracy", lang_correct / len(test_samples))
