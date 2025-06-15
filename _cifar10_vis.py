import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from Multilingual_CLIP.multilingual_clip import pt_multilingual_clip
import clip
from cmcr.cmcr_model import HanCLIP, ModalityType
from safetensors.torch import load_file
import matplotlib as mpl
from PIL import Image

# Set up
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 30

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()


dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=256, shuffle=False)
korean_labels = [
        "비행기", "자동차", "새", "고양이", "사슴",
        "개", "개구리", "말", "배", "트럭"
    ]

# Load models
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

koclip = AutoModel.from_pretrained("koclip/koclip-base-pt").to(device).eval()
koclip_processor = AutoProcessor.from_pretrained("koclip/koclip-base-pt")

mclip = pt_multilingual_clip.MultilingualCLIP.from_pretrained("M-CLIP/XLM-Roberta-Large-Vit-B-32")
mclip.to(device)
mclip.eval()

text_processor = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
hanclip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")


hanclip = HanCLIP('minilm', "korean_embedding_MiniLM_MSCOCO.pt", "image_embedding_CLIP_imagenet.pt")
hanclip_checkpoint = load_file("results/minilm_noise_final/checkpoint-17600/model.safetensors")
hanclip.load_state_dict(hanclip_checkpoint, strict=False)
hanclip.to(device)
hanclip.eval()

# Embedding extraction
def extract_image_embeddings(model_name):
    embeddings = []
    labels = []
    for images, lbls in tqdm(loader, desc=f"Extracting {model_name}"):
        with torch.no_grad():
            if model_name == "CLIP":
                imgs = torch.stack([clip_preprocess(transforms.ToPILImage()(img)) for img in images]).to(device)
                embs = clip_model.encode_image(imgs).cpu().numpy()
            elif model_name == "KoCLIP":
                inputs = koclip_processor(images=[transforms.ToPILImage()(img) for img in images], return_tensors="pt").to(device)
                embs = koclip.get_image_features(**inputs).cpu().numpy()
            # elif model_name == "MCLIP":
            #     imgs = [transform(transforms.ToPILImage()(img)).to(device) for img in images]
            #     pil_imgs = [transforms.ToPILImage()(img.cpu()) for img in imgs]
            #     inputs = torch.stack([clip_preprocess(pil_img) for pil_img in pil_imgs]).to(device)
            #     embs = mclip.vision_model(inputs).cpu().numpy()
            elif model_name == "HanCLIP":
                images_pil = [to_pil(img.cpu()) for img in images]
                vision_inputs = hanclip_processor(images=images_pil, return_tensors="pt").to(device)

                vision_input = {"pixel_values": vision_inputs["pixel_values"]}
                vision_features = hanclip.trunk.get_vision_feature(vision_input)
                projected = hanclip.project_features({ModalityType.VISION: vision_features})
                embs = projected[ModalityType.VISION].cpu().numpy()
        embeddings.append(embs)
        labels.extend(lbls.numpy())
    return np.vstack(embeddings), np.array(labels)

# Collect and plot
for model_name in [ "CLIP","KoCLIP","HanCLIP"]:
    embs, lbls = extract_image_embeddings(model_name)
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(embs)

    plt.figure(figsize=(12, 12))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=lbls, cmap='tab10', s=15)
    # plt.legend(*scatter.legend_elements(), title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.title(f"{model_name} Image Embeddings (CIFAR-10)")
    plt.tight_layout()
    plt.savefig(f"/home/aikusrv01/C-MCR/visualization/cifar10_umap_{model_name.lower()}.png")
    plt.close()
