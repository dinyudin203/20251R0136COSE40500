from datasets import load_dataset
from cmcr.cmcr_model import HanCLIP, ModalityType, MCRType
import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel, AutoProcessor
from PIL import Image, UnidentifiedImageError

from safetensors.torch import load_file
import numpy as np
from itertools import islice
import io
from Multilingual_CLIP.multilingual_clip import pt_multilingual_clip
import clip
from torchvision import transforms
from tqdm import tqdm
from io import BytesIO
import requests
from huggingface_hub import login




device = 'cuda:4' if torch.cuda.is_available() else 'cpu'


transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),  # CLIP 모델 크기
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()


def compute_recall_at_k(image_features, text_features, captions_per_image=5, k_list=[1, 5, 10]):
    """
    image_features: (N, D) tensor, normalized
    text_features: (N*C, D) tensor, normalized
    captions_per_image: how many captions per image (C)
    """
    num_images = image_features.shape[0]
    num_captions = text_features.shape[0]
    assert num_captions == num_images * captions_per_image

    # (N, N*C) similarity matrix
    sims = image_features @ text_features.T

    recalls_image_to_text = {f"Recall@{k}": 0.0 for k in k_list}

    for img_idx in range(num_images):
        # 정답 텍스트 인덱스들
        gt_caption_indices = list(range(img_idx * captions_per_image, (img_idx + 1) * captions_per_image))
        ranking = torch.argsort(sims[img_idx], descending=True)

        # rank는 top-k 안에 정답 캡션 중 하나라도 있으면 성공
        for k in k_list:
            top_k = ranking[:k].tolist()
            if any(gt in top_k for gt in gt_caption_indices):
                recalls_image_to_text[f"Recall@{k}"] += 1

    for k in k_list:
        recalls_image_to_text[f"Recall@{k}"] /= num_images

    # Text-to-Image: 각 캡션마다 정답 이미지가 1개
    sims_text_to_image = text_features @ image_features.T
    recalls_text_to_image = {f"Recall@{k}": 0.0 for k in k_list}

    for cap_idx in range(num_captions):
        gt_image_idx = cap_idx // captions_per_image
        ranking = torch.argsort(sims_text_to_image[cap_idx], descending=True)
        rank = (ranking == gt_image_idx).nonzero(as_tuple=True)[0].item() + 1

        for k in k_list:
            if rank <= k:
                recalls_text_to_image[f"Recall@{k}"] += 1

    for k in k_list:
        recalls_text_to_image[f"Recall@{k}"] /= num_captions

    return recalls_image_to_text, recalls_text_to_image

# Load the dataset
streamed_dataset = load_dataset("jp1924/KoCC3M", split="validation", streaming=True, trust_remote_code=True)
# samples = list(islice(streamed_dataset, 10000))
samples = [x for x in tqdm(islice(streamed_dataset, 10000))]

# Load the model
text_processor = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
# text_processor = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
koclip_processor = AutoProcessor.from_pretrained("koclip/koclip-base-pt")


hanclip = HanCLIP('minilm', "korean_embedding_MiniLM_MSCOCO.pt", "image_embedding_CLIP_imagenet.pt")
hanclip_checkpoint = load_file("results/minilm_noise_final/checkpoint-17600/model.safetensors")
hanclip.load_state_dict(hanclip_checkpoint, strict=False)
hanclip.to(device)
hanclip.eval()

koclip = AutoModel.from_pretrained("koclip/koclip-base-pt")
koclip.to(device)
koclip.eval()


# Load multilingual clip
mclip = pt_multilingual_clip.MultilingualCLIP.from_pretrained("M-CLIP/XLM-Roberta-Large-Vit-B-32")
tokenizer = AutoTokenizer.from_pretrained("M-CLIP/XLM-Roberta-Large-Vit-B-32")
clip_model, preprocess = clip.load("ViT-B/32", device=device)
mclip.eval()
breakpoint()


images = []
texts = []
for i in range(len(samples)):
    image = samples[i]['image'].convert("RGB")
    text = samples[i]['caption']
    images.append(image)
    texts.extend(text)

print(f"Number of images: {len(images)}")
print(f"Number of texts: {len(texts)}")
    
kor_text_embs_hanclip_full = []
img_embs_hanclip_full = []
kor_text_embs_koclip_full = []
img_embs_koclip_full = []
kor_text_embs_mclip_full =[]
img_embs_mclip_full =[]

for i in range(0, len(images), 128):
    images_batch = images[i:i+128]
    if len(images_batch) == 0:
        continue  # 빈 배치는 스킵
    texts_batch = texts[5*i:5*i+len(images_batch)*5]
    if len(texts_batch) != len(images_batch) * 5:
        print(f"[Warning] texts_batch length mismatch at i={i}: got {len(texts_batch)} vs expected {len(images_batch)*5}")
        continue

    # for e5
    # texts_batch = ['query: ' + text for text in texts_batch]

    # Flatten the inputs
    flattened = {}
    # breakpoint()
    clip_vision_inputs = koclip_processor(images=images_batch, return_tensors="pt").to(device)
    kor_inputs = text_processor(texts_batch, return_tensors="pt", padding=True, truncation=True).to(device)
    for k, v in clip_vision_inputs.items():
        flattened[f"clip_vision_{k}"] = v
    for k, v in kor_inputs.items():
        flattened[f"kor_{k}"] = v
    # 임베딩 추출
    with torch.no_grad():
        outputs = hanclip.get_test_embeddings(flattened)
        kor_text_embs_hanclip, img_embs_hanclip = outputs[ModalityType.KOR_TEXT], outputs[ModalityType.VISION]
    with torch.no_grad():
        kor_text_embs_koclip = koclip.get_text_features(**koclip_processor(text=texts_batch, return_tensors="pt", padding=True, truncation=True).to(device))
        img_embs_koclip = koclip.get_image_features(**koclip_processor(images=images_batch, return_tensors="pt").to(device))
    with torch.no_grad():
        text_embeddings = mclip.forward(texts_batch, tokenizer).to(device)
        images = [transform(image) for image in images_batch]
        images_pil = [to_pil(img.cpu()) for img in images]
        image_input = torch.stack([preprocess(pil_img) for pil_img in images_pil]).to(device) 
        image_embeddings = clip_model.encode_image(image_input).float()


    
    
    kor_text_embs_hanclip_full.append(kor_text_embs_hanclip.to('cpu'))
    img_embs_hanclip_full.append(img_embs_hanclip.to('cpu'))
    kor_text_embs_koclip_full.append(kor_text_embs_koclip.to('cpu'))
    img_embs_koclip_full.append(img_embs_koclip.to('cpu'))
    kor_text_embs_mclip_full.append(text_embeddings.to('cpu'))
    img_embs_mclip_full.append(image_embeddings.to('cpu'))

# Concatenate the results
kor_text_embs_hanclip_full = torch.cat(kor_text_embs_hanclip_full, dim=0).to(device)
img_embs_hanclip_full = torch.cat(img_embs_hanclip_full, dim=0).to(device)
kor_text_embs_koclip_full = torch.cat(kor_text_embs_koclip_full, dim=0).to(device)
img_embs_koclip_full = torch.cat(img_embs_koclip_full, dim=0).to(device)
kor_text_embs_mclip_full = torch.cat(kor_text_embs_mclip_full, dim=0).to(device)
img_embs_mclip_full = torch.cat(img_embs_mclip_full, dim=0).to(device)

print("HanCLIP Recall@1,5,10", compute_recall_at_k(img_embs_hanclip_full, kor_text_embs_hanclip_full))
print("KoCLIP Recall@1,5,10", compute_recall_at_k(img_embs_koclip_full, kor_text_embs_koclip_full))
print("MCLIP Recall@1,5,10", compute_recall_at_k(img_embs_mclip_full, kor_text_embs_mclip_full))
