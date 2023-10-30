import os
os.environ["TFHUB_CACHE_DIR"] = '/home/manogna/Cosmos/COSMOS/tfhub_cache'
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.config import *
from utils.text_utils import get_text_metadata
from model_archs.models import CombinedModelMaskRCNN
from utils.common_utils import read_json_data
from utils.eval_utils import is_bbox_overlap, top_bbox_from_scores
from evaluate_ooc import evaluate_context_with_bbox_overlap, get_scores

def load_test_sample(index):

    test_samples = read_json_data(os.path.join(DATA_DIR, 'annotations', 'test_data.json'))
    test_data = test_samples[index]
    return test_data

def draw_bbox(image, bbox, color='red', linewidth=2, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
        ax.imshow(image)
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                             linewidth=linewidth, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    return ax

def visualize_result(test_sample):
    img_path = os.path.join(DATA_DIR, test_sample["img_local_path"])
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pred_context = evaluate_context_with_bbox_overlap(test_sample)
    score_c1, score_c2 = get_scores(test_sample)

    bboxes = test_sample['maskrcnn_bboxes']
    top_bbox_c1 = top_bbox_from_scores(bboxes, score_c1)
    top_bbox_c2 = top_bbox_from_scores(bboxes, score_c2)

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    ax = draw_bbox(image, top_bbox_c1, color='red', ax=ax)
    ax = draw_bbox(image, top_bbox_c2, color='blue', ax=ax)

    if pred_context:
        in_context_caption = "Caption 2 (Blue): " + test_sample['caption2_modified'] + " - In Context"
        out_context_caption = "Caption 1 (Red): " + test_sample['caption1_modified'] + " - Out of Context"
    else:
        in_context_caption = "Caption 1 (Blue): " + test_sample['caption1_modified'] + " - In Context"
        out_context_caption = "Caption 2 (Red): " + test_sample['caption2_modified'] + " - Out of Context"

    plt.figtext(0.1, -0.1, in_context_caption, wrap=True, horizontalalignment='left', fontsize=12, color='blue')
    plt.figtext(0.1, -0.2, out_context_caption, wrap=True, horizontalalignment='left', fontsize=12, color='red')
    plt.axis('off')

    # Save the figure to a file
    output_dir = '/home/manogna/Cosmos/'
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir,"output.pdf")
    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Output image saved to: {output_file_path}")

if __name__ == "__main__":
    index = int(input("Enter the index of the test sample you want to visualize: "))
    test_sample = load_test_sample(index)
    visualize_result(test_sample)