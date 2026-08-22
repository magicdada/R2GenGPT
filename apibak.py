"""
R2GenGPT 推理API服务
使用Delta模型（Epoch 14）生成胸部X光报告
"""
import io
import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from models.R2GenGPT import R2GenGPT
from argparse import Namespace
from transformers import AutoImageProcessor
import numpy as np
from typing import List


# 配置
CHECKPOINT_PATH = "save/iu_xray/v1_delta/checkpoints/checkpoint_epoch14_step7755_bleu0.147965_cider0.413904.pth"
SWIN_MODEL = "microsoft/swin-base-patch4-window7-224"
LLAMA_MODEL = "meta-llama/Llama-2-7b-chat-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="MedReport AI - R2GenGPT Inference API")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型变量
model = None
tokenizer = None
image_processor = None


def load_model():
    """加载模型和预处理器"""
    global model, image_processor

    print("正在加载模型...")

    args = Namespace(
        vision_model=SWIN_MODEL,
        llama_model=LLAMA_MODEL,
        freeze_vm=True,
        vis_use_lora=True,
        vis_r=16,
        vis_alpha=16,
        llm_use_lora=False,
        llm_r=16,
        llm_alpha=16,
        lora_dropout=0.1,
        low_resource=True,
        global_only=False,
        end_sym='</s>',
        savedmodel_path='save/iu_xray/v1_delta',
        ckpt_file=None,
        delta_file=CHECKPOINT_PATH,
        max_length=60,
        beam_size=3,
        do_sample=False,
        no_repeat_ngram_size=2,
        num_beam_groups=1,
        min_new_tokens=40,
        max_new_tokens=100,
        repetition_penalty=2.0,
        length_penalty=2.0,
        diversity_penalty=0,
        temperature=0,
        prompt="",
        img_size=224,
        num_layers=1,
        num_query_token=32,
        weights=[0.5, 0.5],
        scorer_types=['Bleu_4', 'CIDEr'],
        max_epochs=15,
        learning_rate=1e-4,
    )

    model = R2GenGPT(args)
    model.eval()
    model.visual_encoder.to(DEVICE)
    model.llama_proj.to(DEVICE)
    model.layer_norm.to(DEVICE)

    image_processor = AutoImageProcessor.from_pretrained(SWIN_MODEL)

    print("模型加载完成！")


@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    load_model()


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "model": "R2GenGPT-Delta-Epoch14"}



@app.post("/predict")
async def predict(files: List[UploadFile] = File(..., description="上传胸部X光影像（支持多张）")):
    """
    接收胸部X光影像（支持多张），生成诊断报告
    """
    try:
        # 1. 读取所有上传的影像
        pixel_values_list = []
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            # 2. 图像预处理（和训练时data_helper.py保持一致）
            array = np.array(image, dtype=np.uint8)
            if len(array.shape) != 3 or array.shape[-1] != 3:
                array = np.array(image.convert("RGB"), dtype=np.uint8)
            pixel_values = image_processor(array, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(DEVICE)
            pixel_values_list.append(pixel_values)

        # 3. 模型推理
        with torch.no_grad():
            # 编码图像（传入多张，encode_img会取平均）
            img_embeds, atts_img = model.encode_img(pixel_values_list)
            img_embeds = model.layer_norm(img_embeds)
            # 确保所有张量在同一设备上
            img_embeds = img_embeds.to(DEVICE)
            atts_img = atts_img.to(DEVICE)
            # 包装prompt
            img_embeds, atts_img = model.prompt_wrap(img_embeds, atts_img)
            # 转换为float16（匹配4-bit量化LLaMA的精度）
            img_embeds = img_embeds.half()
            # 生成BOS token
            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1],
                             dtype=torch.long,
                             device=img_embeds.device) * model.llama_tokenizer.bos_token_id
            bos_embeds = model.embed_tokens(bos)
            atts_bos = torch.ones([batch_size, 1],
                                  dtype=torch.long,
                                  device=img_embeds.device)
            # 拼接输入
            inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, atts_img], dim=1)
            # 调用LLaMA生成
            outputs = model.llama_model.generate(
                inputs_embeds=inputs_embeds,
                num_beams=model.hparams.beam_size,
                do_sample=model.hparams.do_sample,
                min_new_tokens=model.hparams.min_new_tokens,
                max_new_tokens=model.hparams.max_new_tokens,
                repetition_penalty=model.hparams.repetition_penalty,
                length_penalty=model.hparams.length_penalty,
                temperature=model.hparams.temperature,
            )
            # 解码输出
            report_text = model.decode(outputs[0])

        print(f"生成报告: {report_text}")

        return {
            "report": report_text,
            "heatmap_path": "",
            "status": "success"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "report": "",
            "heatmap_path": "",
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)