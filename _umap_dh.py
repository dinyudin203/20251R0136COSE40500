from datasets import load_dataset
from cmcr.cmcr_model import HanCLIP, ModalityType, MCRType
import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel, AutoProcessor
from PIL import Image
from safetensors.torch import load_file
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from itertools import islice
import io
import torch.nn.functional as F
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

device = 'cuda:4' if torch.cuda.is_available() else 'cpu'

text_processor = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
koclip_processor = AutoProcessor.from_pretrained("koclip/koclip-base-pt")


# Load the dataset
streamed_dataset = load_dataset("kms7530/ko-coco-bal", split="validation", streaming=True, trust_remote_code=True)
samples = list(islice(streamed_dataset, 50))
# Load the model
hanclip = HanCLIP('minilm', "korean_embedding_MiniLM_MSCOCO.pt", "image_embedding_CLIP_imagenet.pt")
hanclip_checkpoint = load_file("results/minilm_noise_final/checkpoint-17600/model.safetensors")
hanclip.load_state_dict(hanclip_checkpoint, strict=False)
hanclip.to(device)
hanclip.eval()


images = []
texts = []
eng = []
for i in range(len(samples)):
    image = samples[i]['image'].convert("RGB")
    text = samples[i]['caption_ko']
    images.append(image)
    texts.append(text)
    eng.append()
    
flattened = {}
clip_vision_inputs = koclip_processor(images=images, return_tensors="pt").to(device)
kor_inputs = text_processor(texts, return_tensors="pt", padding=True, truncation=True).to(device)

for k, v in clip_vision_inputs.items():
    flattened[f"clip_vision_{k}"] = v
for k, v in kor_inputs.items():
    flattened[f"kor_{k}"] = v


with torch.no_grad():
    # projector 통과 전
    clip_input = {
        'pixel_values': flattened["clip_vision_pixel_values"]
    }
    kor_input = {
        'input_ids': flattened["kor_input_ids"],
        'attention_mask': flattened["kor_attention_mask"]
    }

    img_pre = hanclip.trunk.get_vision_feature(clip_input)  # shape: (N, D1)
    text_pre = hanclip.trunk.get_kor_text_feature(kor_input)  # shape: (N, D2)


    # projector 통과 후
    outputs_post = hanclip.get_test_embeddings(flattened)
    img_post = outputs_post[ModalityType.VISION]  # shape: (N, d)
    text_post = outputs_post[ModalityType.KOR_TEXT]  # shape: (N, d)

    # normalize pre for fair comparison (optional)
    img_pre = F.normalize(img_pre, dim=-1)
    text_pre = F.normalize(text_pre, dim=-1)
    
    pca = PCA(n_components=50)
    img_pre_reduced = pca.fit_transform(img_pre.cpu().numpy())
    text_pre_reduced = pca.fit_transform(text_pre.cpu().numpy())

    ###################umap########################

    umap_model_pre = umap.UMAP(n_components=2, random_state=42)
    proj_pre = umap_model_pre.fit_transform(np.concatenate([text_pre_reduced, img_pre_reduced], axis=0))

    # 시각화
    plt.figure(figsize=(8, 6))
    plt.scatter(proj_pre[:len(texts), 0], proj_pre[:len(texts), 1], c='blue', label='Text (Pre)')
    plt.scatter(proj_pre[len(texts):, 0], proj_pre[len(texts):, 1], c='red', label='Image (Pre)')
    plt.legend()
    plt.grid(True)
    plt.title("UMAP: Before Projector")
    plt.savefig("umap_before_projector.png")

    umap_model = umap.UMAP(n_components=2, random_state=42)
    proj = umap_model.fit_transform(np.concatenate([text_post.cpu().numpy(), img_post.cpu().numpy()]))

    # 시각화
    N = len(text_post) 
    text_pts = proj[:N]
    img_pts = proj[N:]

    plt.figure(figsize=(8, 6))
    plt.scatter(text_pts[:, 0], text_pts[:, 1], c='blue', label='Text')
    plt.scatter(img_pts[:, 0], img_pts[:, 1], c='red', label='Image')
    for i in range(N):
        plt.plot([text_pts[i, 0], img_pts[i, 0]], [text_pts[i, 1], img_pts[i, 1]], c='gray', alpha=0.4, linestyle='--')
    plt.title("UMAP with Matching Lines")
    plt.legend()
    plt.grid(True)
    plt.savefig("umap_lines.png")


