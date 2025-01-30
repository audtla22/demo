import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import models, transforms
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from peft import PeftModel
from huggingface_hub import hf_hub_download
import numpy as np
from streamlit_drawable_canvas import st_canvas  # streamlit_drawable_canvas 추가
import cv2

peft_model_ckp = "/home/baiklab/audtla2/이거임"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

@st.cache_resource
def load_models():
    # GPU 0번에서 ResNet101 모델 로드
    device_0 = torch.device("cuda:0")
    checkpoint_path = "/home/baiklab/audtla2/merged/final_model_resnet101_v2.pth"
    resnet101_model = models.resnet101(pretrained=False)
    num_ftrs = resnet101_model.fc.in_features
    resnet101_model.fc = torch.nn.Linear(num_ftrs, 11)
    resnet101_model.load_state_dict(torch.load(checkpoint_path))
    resnet101_model.eval().to(device_0)

    # GPU 1번에서 BLIP2 모델 로드
    device_1 = torch.device("cuda:1")
    blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip2_model = Blip2ForConditionalGeneration.from_pretrained(
        "ybelkada/blip2-opt-2.7b-fp16-sharded", quantization_config=quantization_config
    )
    blip2_model = PeftModel.from_pretrained(blip2_model, peft_model_ckp).to(device_0)

    # Stable Diffusion XL 모델 로드 및 최적화 적용 (GPU 1번)
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_8step_lora.safetensors"  # Use the correct ckpt for your step setting!
    pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=torch.float16, variant="fp16").to(device_1)
    pipe.load_lora_weights(hf_hub_download(repo, ckpt))
    pipe.fuse_lora()
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    
    return resnet101_model, blip2_processor, blip2_model, pipe

def extract_signature(image):
    """서명 이미지를 추출하는 함수 (배경 제거 및 투명 배경 유지)"""
    image = image.convert("RGBA")  # RGBA로 변환 (투명도 포함)
    image_np = np.array(image)  # PIL 이미지를 OpenCV 형식으로 변환
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)  # 그레이스케일 변환

    # 이진화 처리 (배경 제거)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)

    # 투명한 배경을 유지한 채 서명 부분만 추출
    image_np[:, :, 3] = binary  # 알파 채널에 이진화 결과 적용
    signature_pil = Image.fromarray(image_np, mode="RGBA")
    return signature_pil

def add_signature_to_card(card_image, signature_image, position=(400, 300), size=(150, 50)):
    """명함 이미지에 서명을 추가하는 함수"""
    # 서명 이미지 크기 조정
    signature_image = signature_image.resize(size)

    # 명함 이미지를 RGBA로 변환
    card_image = card_image.convert("RGBA")

    # 서명을 명함에 합성 (투명도 유지)
    card_image.paste(signature_image, position, signature_image)
    return card_image

# 모델 로드
resnet101_model, blip2_processor, blip2_model, sdxl_pipeline = load_models()

st.title("AI-Driven Business Card Generator")

# 첫 번째 단계: 이미지 생성
image_data = st.camera_input("Capture from webcam")

if image_data:
    image = Image.open(image_data)
    st.image(image, caption="Captured Image", use_column_width=True)

    preprocess = transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = preprocess(image).unsqueeze(0).to("cuda:0")

    with torch.no_grad():
        outputs = resnet101_model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)

    class_names = ["Casual", "Chic", "Classic", "GoffCore", "Gurlish", "Minimal", "Retro", "Romantic", "Sporty", "Street", "Workwear"]
    style = class_names[predicted_class.item()]
    st.write(f"Detected Style: **{style}**")

    blip_inputs = blip2_processor(images=image, return_tensors="pt").to("cuda:0")  # GPU 1번에 BLIP2 입력
    blip_caption = blip2_model.generate(**blip_inputs)
    caption = blip2_processor.decode(blip_caption[0], skip_special_tokens=True)
    st.write(f"Generated Caption: **{caption}**")

    if 'generated_image' not in st.session_state:
        positive_prompt = (
            f"A stylish background design inspired by {style} fashion. "
            f"The design also reflects color word of the {caption}."
            "Use a sleek, clean layout with subtle textures and elegant color tones that complement the fashion style. "
            "Incorporate abstract geometric patterns and soft gradients. "
            "High resolution, professional, modern, and unique"
        )
        negative_prompt = (
            "low quality, blurry, distorted, cluttered, text, watermark, logo, cartoonish, "
            "overexposed, underexposed, pixelated, overly dark, overly bright, busy patterns, harsh gradients, "
            "realistic photos of people, unnecessary objects, irrelevant scenes"
        )
        width = 1024
        height = 512
        with torch.cuda.device("cuda:1"):
            st.session_state.generated_image = sdxl_pipeline(positive_prompt, height=height, width=width, negative_prompt=negative_prompt, num_inference_steps=8, guidance_scale=7.5).images[0]
    st.image(st.session_state.generated_image, caption="Generated Business Card", use_column_width=True)

# 서명 추가 부분
if 'generated_image' in st.session_state:
    st.write("### Draw Your Signature")
    canvas_result = st_canvas(
        stroke_width=5,
        stroke_color="black",
        background_color="white",
        width=400,
        height=100,
        drawing_mode="freedraw",
        key="signature_canvas",  # 캔버스를 유니크한 키로 설정
    )

    if canvas_result.image_data is not None:
        # 캔버스에서 서명 추출
        signature_image = Image.fromarray(canvas_result.image_data.astype(np.uint8))
        signature_only = extract_signature(signature_image)

        if signature_only:
            # 명함에 서명 추가
            generated_image = st.session_state.generated_image
            updated_card = add_signature_to_card(generated_image, signature_only, position=(400, 300), size=(150, 50))
            st.session_state.generated_image = updated_card  # 서명 추가 후 명함 저장
            st.image(updated_card, caption="Business Card with Signature", use_column_width=True)

# Add text to the card
st.write("### Add Your Text to the Card")
name = st.text_input("Name")
position = st.text_input("Position")
contact = st.text_input("Contact")

# 글자색 선택 (라디오 버튼 방식)
st.write("Select Text Color:")
custom_color = st.color_picker("Pick a custom color", "#000000")
selected_color = custom_color

if st.button("Add Text"):
    generated_image = st.session_state.generated_image
    draw = ImageDraw.Draw(generated_image)
    font_path = "/home/baiklab/audtla2/BMJUA.ttf"  # 예시
    font1 = ImageFont.truetype(font_path, size=60)
    font2 = ImageFont.truetype(font_path, size=40)
    font3 = ImageFont.truetype(font_path, size=20)

    draw.text((200, 350), f"{name}", fill=custom_color, font=font1)
    draw.text((50, 360), f"{position}", fill=custom_color, font=font2)
    draw.text((50, 450), f"Contact: {contact}", fill=custom_color, font=font3)
    st.image(generated_image, caption="Business Card with Text", use_column_width=True)
# Step 4: Download the final card
if 'generated_image' in st.session_state:
    st.write("### Download Your Business Card")
    card_download = st.session_state.generated_image
    card_download.save("business_card.png")

    with open("business_card.png", "rb") as file:
        btn = st.download_button(
            label="Download Business Card",
            data=file,
            file_name="business_card.png",
            mime="image/png"
        )
