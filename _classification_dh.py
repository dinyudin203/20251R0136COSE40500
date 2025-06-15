from torchvision.datasets import CIFAR10, Food101, Caltech101, ImageFolder, OxfordIIITPet, CIFAR100, STL10
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoProcessor, AutoModel
import torch
from sklearn.metrics import classification_report, f1_score
import argparse
import numpy as np
from cmcr.cmcr_model import HanCLIP, ModalityType, MCRType
from safetensors.torch import load_file
import torch.nn.functional as F
from datasets import load_dataset
from Multilingual_CLIP.multilingual_clip import pt_multilingual_clip
import transformers
import clip


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--model", type=str, default=None)         
args = parser.parse_args()


transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),  # CLIP 모델 크기
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()


if args.dataset == "cifar10":
    dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    korean_labels = [
        "비행기", "자동차", "새", "고양이", "사슴",
        "개", "개구리", "말", "배", "트럭"
    ]

elif args.dataset == "stl10":
    dataset = STL10(root='./data', split='test', download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    korean_labels = ["비행기", "새",  "자동차",  "고양이", "사슴", "강아지", "말", "원숭이","배", "트럭"]

elif args.dataset == "food101":
    dataset = Food101(root='./data', split='test', transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    food_classes = [
        "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
        "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
        "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
        "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
        "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
        "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
        "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
        "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
        "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
        "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
        "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
        "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
        "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
        "mussels", "nachos", "omelette", "onion_rings", "oysters",
        "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
        "pho", "pizza", "pork_chop", "poutine", "prime_rib",
        "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
        "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
        "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
        "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"
    ]
    korean_labels = [
        "애플파이", "바비큐 폭립", "바클라바", "비프 카르파초", "육회",
        "비트 샐러드", "베녜", "비빔밥", "브레드 푸딩", "아침 부리토",
        "브루스케타", "시저 샐러드", "카놀리", "카프레제 샐러드", "당근 케이크",
        "세비체", "치즈케이크", "치즈 플래터", "치킨 커리", "치킨 케사디야",
        "치킨 윙", "초콜릿 케이크", "초콜릿 무스", "추로스", "클램 차우더",
        "클럽 샌드위치", "크랩 케이크", "크렘 브륄레", "크로크 마담", "컵케이크",
        "데블드 에그", "도넛", "만두", "에다마메", "에그 베네딕트",
        "에스카르고", "팔라펠", "필레 미뇽", "피시 앤 칩스", "푸아그라",
        "감자튀김", "프렌치 어니언 수프", "프렌치 토스트", "오징어 튀김", "볶음밥",
        "프로나 요거트", "갈릭 브레드", "뇨끼", "그리스 샐러드", "그릴드 치즈 샌드위치",
        "연어구이", "과카몰리", "교자", "햄버거", "매콤 새콤 수프",
        "핫도그", "웨보스 란체로스", "후무스", "아이스크림", "라자냐",
        "랍스터 비스크", "랍스터 롤 샌드위치", "맥앤치즈", "마카롱", "미소된장국",
        "홍합 요리", "나초", "오믈렛", "어니언 링", "굴 요리",
        "팟타이", "파에야", "팬케이크", "판나코타", "북경오리",
        "쌀국수", "피자", "돼지갈비찜", "푸틴", "프라임 립",
        "풀드 포크 샌드위치", "라멘", "라비올리", "레드벨벳 케이크", "리조또",
        "사모사", "사시미", "가리비 요리", "해초 샐러드", "새우와 그리츠",
        "스파게티 볼로네제", "스파게티 카르보나라", "스프링 롤", "스테이크", "딸기 쇼트케이크",
        "스시", "타코", "타코야키", "티라미수", "참치 타르타르", "와플"
    ]
 
elif args.dataset == "caltech101":
    dataset = ImageFolder(root='/home/aikusrv01/C-MCR/data/caltech101/101_ObjectCategories', transform=transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    class_names = [
    'accordion', 'airplanes', 'anchor', 'ant', 'background_google',
    'barrel', 'bass', 'beaver', 'binocular', 'bonsai',
    'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera',
    'cannon', 'car_side', 'ceiling_fan', 'cellphone', 'chair',
    'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish',
    'crocodile', 'crow', 'cup', 'dalmatian', 'dollar_bill',
    'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu',
    'euphonium', 'ewer', 'faces', 'faces_easy', 'ferry',
    'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone',
    'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter',
    'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch',
    'lamp', 'laptop', 'leopards', 'llama', 'lobster',
    'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome',
    'minaret', 'motorbikes', 'nautilus', 'octopus', 'okapi',
    'pagoda', 'panda', 'pigeon', 'pizza', 'platypus',
    'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone',
    'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy',
    'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign',
    'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella',
    'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair',
    'wrench', 'yin_yang'
    ]
    korean_labels = [
        "아코디언", "비행기", "닻", "개미", "배경_구글",
        "통", "농어", "비버", "쌍안경", "분재",
        "뇌", "브론토사우루스", "부처", "나비", "카메라",
        "대포", "자동차_측면", "천장 선풍기", "휴대폰", "의자",
        "샹들리에", "쿠거_몸통", "쿠거_얼굴", "게", "가재",
        "악어", "까마귀", "컵", "달마시안", "달러 지폐",
        "돌고래", "잠자리", "일렉트릭 기타", "코끼리", "에뮤",
        "유포니움", "물병", "얼굴", "쉬운 얼굴", "페리호",
        "홍학", "홍학_머리", "가필드", "게레눅", "축음기",
        "그랜드 피아노", "매부리거북", "헤드폰", "고슴도치", "헬리콥터",
        "따오기", "인라인 스케이트", "여호수아 나무", "캥거루", "케치 요트",
        "램프", "노트북", "표범", "라마", "바닷가재",
        "연꽃", "만돌린", "하루살이", "유대 촛대", "메트로놈",
        "미나렛", "오토바이", "앵무조개", "문어", "오카피",
        "파고다", "판다", "비둘기", "피자", "오리너구리",
        "피라미드", "리볼버", "코뿔소", "수탉", "색소폰",
        "범선", "가위", "전갈", "해마", "스누피",
        "축구공", "스테이플러", "불가사리", "스테고사우루스", "정지 표지판",
        "딸기", "해바라기", "진드기", "삼엽충", "우산",
        "손목시계", "수련", "휠체어", "야생 고양이", "윈저 의자",
        "렌치", "음양"
    ]

elif args.dataset == "oxfordpets" :
    dataset = OxfordIIITPet(root='./data', split='test', target_types='category', download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    class_names = dataset.classes  # 총 37개의 품종 이름
    korean_labels = [
        "아비시니안", "아메리칸 불독", "아메리칸 핏불테리어", "발리니즈", "베이글", "벵골",
        "비즐라", "봄베이", "브리티시 쇼트헤어", "샴", "이집션 마우", "잉글리시 코커 스패니얼",
        "잉글리시 폭스하운드", "이그조틱 쇼트헤어", "그레이하운드", "하바니즈", "히말라얀",
        "저먼 쇼트헤어드 포인터", "그레이트 페레니즈", "재패니즈 친", "케르리 블루 테리어",
        "레오파드", "마인쿤", "맹크스", "미니어처 핀셔", "네벨룽", "노퍽 테리어", "노르웨이 숲 고양이",
        "페르시안", "라가머핀", "러시안 블루", "스코티시 폴드", "쉐틀랜드 쉽독", "시베리안 허스키",
        "싱가푸라", "소말리", "요크셔 테리어"
    ]

elif args.dataset == "cifar100":
    dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    class_names = dataset.classes  
    korean_labels = [
        '사과', '열대어', '아기', '곰', '비버', '침대', '벌', '딱정벌레', '자전거', '병',
        '그릇', '소년', '다리', '버스', '나비', '낙타', '깡통', '성', '애벌레', '소',
        '의자', '침팬지', '시계', '구름', '바퀴벌레', '소파', '게', '악어', '컵', '공룡',
        '돌고래', '코끼리', '넙치', '숲', '여우', '소녀', '햄스터', '집', '캥거루', '키보드',
        '램프', '잔디깎이', '표범', '사자', '도마뱀', '바닷가재', '남자', '단풍나무', '오토바이', '산',
        '쥐', '버섯', '참나무', '오렌지', '난초', '수달', '야자수', '배', '픽업트럭', '소나무',
        '평야', '접시', '양귀비', '호저', '주머니쥐', '토끼', '라쿤', '가오리', '도로', '로켓',
        '장미', '바다', '물개', '상어', '땃쥐', '스컹크', '고층빌딩', '달팽이', '뱀', '거미',
        '다람쥐', '노면전차', '해바라기', '피망', '탁자', '탱크', '전화기', '텔레비전', '호랑이', '트랙터',
        '기차', '송어', '튤립', '거북이', '옷장', '고래', '버드나무', '늑대', '여자', '지렁이'
    ]

with torch.no_grad():
    if args.model == "koclip" :
        model = AutoModel.from_pretrained("koclip/koclip-base-pt").eval().cuda()
        processor = AutoProcessor.from_pretrained("koclip/koclip-base-pt")

        text_inputs = processor(text=korean_labels, return_tensors="pt", padding=True).to("cuda")
        text_features = model.get_text_features(**text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        all_preds = []
        all_labels = []

        if args.dataset in ["cifar10" , "cifar100","food101", "caltech101", "oxfordpets", "stl10"] :
            with torch.no_grad():
                for images, labels in loader:
                    images = images.cuda()
                    inputs = processor(images=images, return_tensors="pt", do_rescale=False).to("cuda")
                    image_features = model.get_image_features(**inputs)
                    image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

                    logits = image_features @ text_features.T
                    preds = logits.argmax(dim=1).cpu()

                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels.tolist())

            # macro: 클래스마다 F1 계산 후 평균
            # weighted: 클래스별 샘플 수를 고려한 평균
            print(classification_report(all_labels, all_preds, target_names=korean_labels))
            macro_f1 = f1_score(all_labels, all_preds, average='macro')
            weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

            print(f"{args.dataset}-{args.model} Macro F1 Score: {macro_f1:.4f}")
            print(f"{args.dataset}-{args.model} Weighted F1 Score: {weighted_f1:.4f}")
        


    elif args.model == "hanclip" :
        text_processor = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        koclip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        hanclip = HanCLIP('minilm', "korean_embedding_MiniLM_MSCOCO.pt", "image_embedding_CLIP_imagenet.pt")
        hanclip_checkpoint = load_file("results/minilm_noise_final/checkpoint-17600/model.safetensors")
        hanclip.load_state_dict(hanclip_checkpoint, strict=False)
        hanclip.to(device)
        hanclip.eval()


        flattened={}
        text_inputs = text_processor(text=korean_labels, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            kor_input = {
                'input_ids': text_inputs['input_ids'],
                'attention_mask': text_inputs['attention_mask']
            }
            kor_features = hanclip.trunk.get_kor_text_feature(kor_input)
            projected = hanclip.project_features({ModalityType.KOR_TEXT: kor_features})
            text_embeds = projected[ModalityType.KOR_TEXT]
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        all_preds, all_labels = [], []
        if args.dataset in ["cifar10" , "cifar100","food101", "caltech101", "oxfordpets", "stl10"] :
            with torch.no_grad():
                for images, labels in loader:
                    images_pil = [to_pil(img.cpu()) for img in images]
                    vision_inputs = koclip_processor(images=images_pil, return_tensors="pt").to(device)

                    vision_input = {"pixel_values": vision_inputs["pixel_values"]}
                    vision_features = hanclip.trunk.get_vision_feature(vision_input)
                    projected = hanclip.project_features({ModalityType.VISION: vision_features})
                    image_embeds = projected[ModalityType.VISION]
                    image_embeds = F.normalize(image_embeds, p=2, dim=-1)

                    logits = image_embeds @ text_embeds.T
                    preds = logits.argmax(dim=1).cpu()

                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels.tolist())
            print(classification_report(all_labels, all_preds, target_names=korean_labels))
            macro_f1 = f1_score(all_labels, all_preds, average='macro')
            weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

            print(f"{args.dataset}-{args.model} Macro F1 Score: {macro_f1:.4f}")
            print(f"{args.dataset}-{args.model} Weighted F1 Score: {weighted_f1:.4f}")

    elif args.model == "multilingual_clip":
        model = pt_multilingual_clip.MultilingualCLIP.from_pretrained("M-CLIP/XLM-Roberta-Large-Vit-B-32")
        tokenizer = AutoTokenizer.from_pretrained("M-CLIP/XLM-Roberta-Large-Vit-B-32")
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        text_embeddings = model.forward(korean_labels, tokenizer).to(device)
        all_preds = []
        all_labels = []

        if args.dataset in ["cifar10" , "cifar100","food101", "caltech101", "oxfordpets", "stl10"] :
            with torch.no_grad():
                for images, labels in loader:
                    
                    images_pil = [to_pil(img.cpu()) for img in images]
                    image_input = torch.stack([preprocess(pil_img) for pil_img in images_pil]).to(device) 
                    image_features = clip_model.encode_image(image_input)
                    image_embeddings = image_features.float()
                    text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
                    image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)

                    # inputs = processor(images=images, return_tensors="pt", do_rescale=False).to("cuda")
                    # image_features = model.get_image_features(**inputs)
                    # image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

                    logits = image_embeddings @ text_embeddings.T
                    preds = logits.argmax(dim=1).cpu()

                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels.tolist())

            # macro: 클래스마다 F1 계산 후 평균
            # weighted: 클래스별 샘플 수를 고려한 평균
            print(classification_report(all_labels, all_preds, target_names=korean_labels))
            macro_f1 = f1_score(all_labels, all_preds, average='macro')
            weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

            print(f"{args.dataset}-{args.model} Macro F1 Score: {macro_f1:.4f}")
            print(f"{args.dataset}-{args.model} Weighted F1 Score: {weighted_f1:.4f}")



