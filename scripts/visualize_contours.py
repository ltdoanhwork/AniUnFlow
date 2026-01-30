import os
import matplotlib.pyplot as plt
from PIL import Image
import random
import glob
import sys
import numpy as np

# Add project root to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.flow_viz import flow_to_image

def read_flow(filename):
    """
    read optical flow from Middlebury .flo file
    :param filename: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        # print("Reading %d x %d flo file" % (h, w))
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d

def visualize_samples(original_dir, contour_dir, flow_dir, output_path="contour_flow_visualization.png", num_samples=5):
    """
    Visualizes random samples of original images, contours, and flow ground truth.
    """
    
    # Get list of files
    original_files = sorted(glob.glob(os.path.join(original_dir, "*.png")))
    
    if not original_files:
        print(f"No images found in {original_dir}")
        return

    # Select random samples
    # Ensure corresponding contour and flow exists
    valid_samples = []
    for f in original_files:
        basename = os.path.basename(f)
        contour_path = os.path.join(contour_dir, basename)
        flow_path = os.path.join(flow_dir, basename.replace(".png", ".flo"))
        
        if os.path.exists(contour_path) and os.path.exists(flow_path):
            valid_samples.append(basename)
            
    if not valid_samples:
        print(f"No matching contour/flow files found in {contour_dir} and {flow_dir}")
        return

    selected_samples = sorted(random.sample(valid_samples, min(num_samples, len(valid_samples))))
    
    print(f"Visualizing samples: {selected_samples}")

    fig, axes = plt.subplots(3, len(selected_samples), figsize=(4 * len(selected_samples), 9))
    
    if len(selected_samples) == 1:
        axes = axes.reshape(3, 1)

    for i, sample_name in enumerate(selected_samples):
        orig_path = os.path.join(original_dir, sample_name)
        cont_path = os.path.join(contour_dir, sample_name)
        fl_path = os.path.join(flow_dir, sample_name.replace(".png", ".flo"))
        
        orig_img = Image.open(orig_path).convert("RGB")
        cont_img = Image.open(cont_path).convert("RGB") 
        
        flow_data = read_flow(fl_path)
        flow_img = flow_to_image(flow_data)
        
        # Original Image
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f"Original: {sample_name}")
        axes[0, i].axis('off')
        
        # Contour Image
        axes[1, i].imshow(cont_img)
        axes[1, i].set_title(f"Contour")
        axes[1, i].axis('off')

        # Flow Image
        axes[2, i].imshow(flow_img)
        axes[2, i].set_title(f"Flow")
        axes[2, i].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    original_dir = "/home/serverai/ltdoanh/AniUnFlow/data/AnimeRun_v2/train/Frame_Anime/agent_indoor6_push_button/original"
    contour_dir = "/home/serverai/ltdoanh/AniUnFlow/data/AnimeRun_v2/train/contour/agent_indoor6_push_button"
    flow_dir = "/home/serverai/ltdoanh/AniUnFlow/data/AnimeRun_v2/train/Flow/agent_indoor6_push_button/forward"
    
    visualize_samples(original_dir, contour_dir, flow_dir)

