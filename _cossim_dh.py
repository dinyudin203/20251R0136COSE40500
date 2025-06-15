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
from Multilingual_CLIP.multilingual_clip import pt_multilingual_clip
import clip
from torchvision import transforms



device = 'cuda:4' if torch.cuda.is_available() else 'cpu'


transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),  # CLIP 모델 크기
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()


text_processor = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
koclip_processor = AutoProcessor.from_pretrained("koclip/koclip-base-pt")


koclip = AutoModel.from_pretrained("koclip/koclip-base-pt")
koclip.to(device)
koclip.eval()


# Load the dataset
streamed_dataset = load_dataset("kms7530/ko-coco-bal", split="validation", streaming=True, trust_remote_code=True)
samples = list(islice(streamed_dataset, 50))
# Load the model
hanclip = HanCLIP('minilm', "korean_embedding_MiniLM_MSCOCO.pt", "image_embedding_CLIP_imagenet.pt")
hanclip_checkpoint = load_file("results/minilm_noise_final/checkpoint-17600/model.safetensors")
hanclip.load_state_dict(hanclip_checkpoint, strict=False)
hanclip.to(device)
hanclip.eval()
# Load multilingual clip
mclip = pt_multilingual_clip.MultilingualCLIP.from_pretrained("M-CLIP/XLM-Roberta-Large-Vit-B-32")
tokenizer = AutoTokenizer.from_pretrained("M-CLIP/XLM-Roberta-Large-Vit-B-32")
clip_model, preprocess = clip.load("ViT-B/32", device=device)



images = []
texts = []
for i in range(len(samples)):
    image = samples[i]['image'].convert("RGB")
    text = samples[i]['caption_ko']
    images.append(image)
    texts.append(text)
    
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

    # koclip
    kor_text_embs_koclip = koclip.get_text_features(**koclip_processor(texts, return_tensors="pt", padding=True, truncation=True).to(device))
    img_embs_koclip = koclip.get_image_features(**koclip_processor(images=images, return_tensors="pt").to(device))
    kor_text_embs_koclip = F.normalize(kor_text_embs_koclip, dim=-1).cpu().numpy()
    img_embs_koclip = F.normalize(img_embs_koclip, dim=-1).cpu().numpy()

    # mclip
    text_embeddings = mclip.forward(texts, tokenizer).to(device)
    # breakpoint()
    images = [transform(image) for image in images]
    images_pil = [to_pil(img.cpu()) for img in images]
    image_input = torch.stack([preprocess(pil_img) for pil_img in images_pil]).to(device) 
    image_embeddings = clip_model.encode_image(image_input).float()
    text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)



    ####################cosine sim#########################
    # normalize pre for fair comparison (optional)
    img_pre = F.normalize(img_pre, dim=-1)
    text_pre = F.normalize(text_pre, dim=-1)
    
    pca = PCA(n_components=50)
    img_pre_reduced = pca.fit_transform(img_pre.cpu().numpy())
    text_pre_reduced = pca.fit_transform(text_pre.cpu().numpy())
    # cos sim
    sim_pre = cosine_similarity(text_pre_reduced, img_pre_reduced)
    sim_post = cosine_similarity(text_post.cpu().numpy(), img_post.cpu().numpy())
    sim_koclip = cosine_similarity(kor_text_embs_koclip, img_embs_koclip)
    sim_mclip = cosine_similarity(text_embeddings.cpu().numpy(), image_embeddings.cpu().numpy())
    # heatmap 시각화

    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 70

    sim_matrices = [sim_pre, sim_post, sim_koclip, sim_mclip]
    titles = ["before_projector", "after_projector", "koclip", "mclip"]
    save_dir = "/home/aikusrv01/C-MCR/visualization/"

    for sim_matrix, title in zip(sim_matrices, titles):
        fig, ax = plt.subplots(figsize=(15, 15))  # 개별 그림 생성

        sns.heatmap(sim_matrix, cmap='coolwarm', ax=ax, cbar=False)

        # 축 라벨
        ax.set_xlabel("Text Index", fontsize=70)
        ax.set_ylabel("Image Index", fontsize=70)

        # tick 간격 10으로 설정
        tick_locs = np.arange(0, sim_matrix.shape[0], 10)
        ax.set_xticks(tick_locs)
        ax.set_yticks(tick_locs)

        ax.tick_params(axis='x', labelsize=70)
        ax.tick_params(axis='y', labelsize=70)

        plt.tight_layout()
        plt.savefig(f"{save_dir}cosine_similarity_{title}.png")
        plt.close()




