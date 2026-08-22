"""
R2GenGPT 推理API服务
胸部X光报告生成 + 遮挡式病灶热力图（正位/侧位分别出图）

热力图方法: 保留式遮挡（因果扰动）
    只保留影像的一小块、其余置为数据集均值, 用 teacher forcing 重算已生成报告的
    log-likelihood。某块单独能把报告恢复得越多, 说明模型说出这段话越依赖它。

    测的是因果依赖而非模型内部中间量, 所以不受注意力伪影影响。
    之前用 LLaMA 跨模态注意力做过一版, 实测存在按 token 序号周期交替的排布伪影,
    且不同患者之间热力图结构几乎一致, 已弃用。

显示闸门（重要）:
    只有报告里抽到阳性病名时才出词级热力图。阴性报告不出图 ——
    热力图经过归一化后永远有红色区域, 正常片上展示会被误读成病灶定位。

依赖: fastapi uvicorn torch transformers pillow numpy
"""
import base64
import io
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import uvicorn
from argparse import Namespace
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoImageProcessor

from models.R2GenGPT import R2GenGPT

# 配置
CHECKPOINT_PATH = "save/iu_xray/v1_delta/checkpoints/checkpoint_epoch14_step7755_bleu0.147965_cider0.413904.pth"
SWIN_MODEL = "microsoft/swin-base-patch4-window7-224"
LLAMA_MODEL = "meta-llama/Llama-2-7b-chat-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENABLE_HEATMAP = os.getenv("ENABLE_HEATMAP", "1") == "1"
GRID = int(os.getenv("HEATMAP_GRID", "7"))        # 遮挡网格, 与视觉 token 的 7x7 对齐
BATCH = int(os.getenv("HEATMAP_BATCH", "4"))      # 批大小, OOM 会自动退回 1
VIEW_NAMES = ["frontal", "lateral"]               # 按上传顺序命名

app = FastAPI(title="MedReport AI - R2GenGPT Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
image_processor = None
explainer = None


# 报告文本解析
CHEXPERT_LEXICON: Dict[str, List[str]] = {
    "Atelectasis": [r"atelecta\w*"],
    "Cardiomegaly": [r"cardiomegaly", r"enlarged?\s+heart", r"cardiac\s+enlargement",
                     r"heart\s+size\s+is\s+(?:mildly\s+)?enlarged"],
    "Consolidation": [r"consolidat\w*"],
    "Edema": [r"edema"],
    "Enlarged Cardiomediastinum": [r"cardiomediastin\w*", r"mediastinal\s+widening"],
    "Fracture": [r"fractur\w*"],
    "Lung Lesion": [r"nodul\w*", r"\blesion\w*", r"\bmass\b", r"masses"],
    "Lung Opacity": [r"opacit\w*", r"opacification", r"infiltrat\w*"],
    "Pleural Effusion": [r"effusion\w*"],
    "Pleural Other": [r"pleural\s+thickening", r"pleural\s+scarring"],
    "Pneumonia": [r"pneumonia"],
    "Pneumothorax": [r"pneumothora\w*"],
    "Support Devices": [r"catheter", r"\btube\b", r"pacemaker", r"\bport\b", r"picc",
                        r"sternotomy\s+wire\w*"],
    "Emphysema": [r"emphysema\w*", r"hyperinflat\w*"],
    "Scarring": [r"scarring", r"fibro\w*"],
    "Calcification": [r"calcifi\w*", r"granuloma\w*"],
}

# 否定线索。不能用固定字符窗口: 报告里常见一个否定词管一串病名
# ("no focal consolidation pneumothorax or large pleural effusion"), 窗口开小了会漏
# 掉最后一个。按句子边界判断, 不设长度上限。
_NEG_CUE = re.compile(
    r"\b(no|not|without|free\s+of|absence\s+of|negative\s+for|clear\s+of|"
    r"unremarkable|resolution\s+of|rule\s+out|denies)\b", re.IGNORECASE)

# 正常性表述。这类是后置修饰, 出现在病名"后面"
# ("the cardiomediastinal silhouette ... are within normal limits"),
# 所以整句任意位置命中都要排除, 只查前缀不够。
_NORMAL_CUE = re.compile(
    r"within\s+normal\s+limits|(?:is|are|appears?)\s+(?:grossly\s+)?normal|"
    r"normal\s+in\s+(?:size|caliber|configuration|appearance)|unremarkable|"
    r"grossly\s+normal|no\s+evidence\s+of", re.IGNORECASE)

# 词表覆盖不到但明显在描述异常的措辞。命中说明报告不是纯阴性,
# 只是无法归到具体病名 —— 这种情况不能当正常处理。
_ABNORMAL_CUE = re.compile(
    r"\babnormal\w*|increased?|decreased?|prominen\w*|suspicious|concerning|"
    r"worsen\w*|progress\w*|irregular|blunt\w*|elevat\w*|deviat\w*|\bnew\b|"
    r"interval\s+change|ill[-\s]defined|asymmetr\w*", re.IGNORECASE)

_SENT_END = ".;\n"
_PATTERNS = {k: [re.compile(p, re.IGNORECASE) for p in v]
             for k, v in CHEXPERT_LEXICON.items()}


def _sentence(text: str, pos: int) -> Tuple[int, int]:
    start = max(text.rfind(ch, 0, pos) for ch in _SENT_END) + 1
    ends = [e for e in (text.find(ch, pos) for ch in _SENT_END) if e >= 0]
    return start, (min(ends) if ends else len(text))


def is_negated(text: str, kw_start: int) -> bool:
    """病名是否处于阴性语境: 同句中前面有否定词, 或整句有正常性表述"""
    s, e = _sentence(text, kw_start)
    return bool(_NEG_CUE.search(text[s:kw_start]) or _NORMAL_CUE.search(text[s:e]))


def find_keywords(text: str, limit: int = 5) -> List[Dict]:
    hits = []
    for label, patterns in _PATTERNS.items():
        for pat in patterns:
            matched = False
            for m in pat.finditer(text):
                if is_negated(text, m.start()):
                    continue
                hits.append({"label": label, "keyword": m.group(0),
                             "span": (m.start(), m.end())})
                matched = True
                break
            if matched:
                break
    hits.sort(key=lambda h: h["span"][0])
    return hits[:limit]


def assess(text: str, hits: List[Dict]) -> Tuple[str, str]:
    """
    findings  — 抽到阳性病名, 展示词级热力图
    uncertain — 无病名但有未归类的异常措辞, 展示整体图并标注
    normal    — 纯阴性, 不展示任何热力图
    """
    if hits:
        return "findings", f"抽到 {len(hits)} 个阳性病名"
    for m in _ABNORMAL_CUE.finditer(text):
        if not is_negated(text, m.start()):
            return "uncertain", f"未归类的异常措辞: {m.group(0)}"
    return "normal", "未检出阳性描述"


# 病名 -> Impression 用语。用规范表述，不直接抄 Findings 的原文
IMPRESSION_TERMS: Dict[str, str] = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "airspace consolidation",
    "Edema": "pulmonary edema",
    "Enlarged Cardiomediastinum": "widened cardiomediastinal silhouette",
    "Fracture": "fracture",
    "Lung Lesion": "pulmonary nodule/lesion",
    "Lung Opacity": "pulmonary opacity",
    "Pleural Effusion": "pleural effusion",
    "Pleural Other": "pleural thickening",
    "Pneumonia": "findings suspicious for pneumonia",
    "Pneumothorax": "pneumothorax",
    "Support Devices": "indwelling support device",
    "Emphysema": "pulmonary hyperinflation, suggesting emphysema",
    "Scarring": "pulmonary scarring/fibrosis",
    "Calcification": "calcified granuloma",
}


def build_impression(status: str, hits: List[Dict]) -> str:
    """
    从 Findings 里抽到的病名生成 Impression 草稿。

    不能写死: Impression 是临床医生真正阅读的结论段, 写死会与模型生成的 Findings
    直接矛盾(阳性 Findings 配"未见异常"的 Impression)。这里由同一批抽取结果推导,
    两段天然一致。

    这只是把模型自己的输出归纳成结论, 不引入新判断, 必须经医师复核后定稿。
    """
    if status == "findings" and hits:
        terms = []
        for h in hits:
            t = IMPRESSION_TERMS.get(h["label"], h["label"].lower())
            if t not in terms:
                terms.append(t)
        if len(terms) == 1:
            return terms[0].capitalize() + "."
        return "\n".join(f"{i}. {t.capitalize()}." for i, t in enumerate(terms, 1))

    if status == "uncertain":
        return ("Findings as described above; no specific diagnosis assigned. "
                "Recommend correlation with clinical history.")

    return "No acute cardiopulmonary abnormality."


# 渲染（纯 numpy + PIL）
def _jet(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _normalize(m: np.ndarray) -> np.ndarray:
    lo, hi = float(m.min()), float(m.max())
    if hi - lo < 1e-8:
        return np.zeros_like(m, dtype=np.float32)
    return ((m - lo) / (hi - lo)).astype(np.float32)


def render_overlay(image: Image.Image, grid_map: np.ndarray, out_size: int = 448,
                   alpha: float = 0.5, floor: float = 0.15) -> str:
    """把网格热力图叠到原图, 返回 base64 PNG（不含 data URI 前缀）"""
    base = image.convert("RGB")
    w, h = base.size
    scale = out_size / max(w, h)
    if scale < 1.0:
        base = base.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    w, h = base.size

    heat = _normalize(grid_map)
    mid = Image.fromarray((heat * 255).astype(np.uint8), mode="L")
    mid = mid.resize((heat.shape[1] * 8, heat.shape[0] * 8), Image.BICUBIC)
    heat = np.asarray(mid.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0

    a = (np.clip((heat - floor) / max(1e-6, 1.0 - floor), 0.0, 1.0) * alpha)[..., None]
    blended = (np.asarray(base, dtype=np.float32) * (1.0 - a)
               + _jet(heat).astype(np.float32) * a).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# 遮挡式热力图
class OcclusionExplainer:

    def __init__(self, model, image_processor, device: str = "cuda",
                 grid: int = 7, batch_size: int = 4, out_size: int = 448):
        self.model = model
        self.image_processor = image_processor
        self.device = device
        self.grid = grid
        self.batch_size = batch_size
        self.out_size = out_size

    def to_pixel_values(self, image: Image.Image) -> torch.Tensor:
        """与训练时 data_helper.py 保持一致"""
        array = np.array(image, dtype=np.uint8)
        if len(array.shape) != 3 or array.shape[-1] != 3:
            array = np.array(image.convert("RGB"), dtype=np.uint8)
        pv = self.image_processor(array, return_tensors="pt").pixel_values
        return pv.to(self.device)

    def _occlude(self, pv: torch.Tensor, cells) -> torch.Tensor:
        """
        遮住指定格子。pixel_values 已标准化, 填 0 等价于填数据集均值 ——
        最中性的基线, 不会引入新的边缘或亮度线索。
        """
        if isinstance(cells, (int, np.integer)):
            cells = [int(cells)]
        side, out = self.grid, pv.clone()
        h, w = pv.shape[-2], pv.shape[-1]
        for cell in cells:
            r, c = divmod(int(cell), side)
            out[..., round(r * h / side):round((r + 1) * h / side),
                round(c * w / side):round((c + 1) * w / side)] = 0.0
        return out

    @torch.no_grad()
    def _logprobs(self, pv_slots: List[torch.Tensor], gen_ids: torch.Tensor) -> np.ndarray:
        """一次前向算出每个 token 的 log-prob, 返回 [B, T]"""
        m = self.model
        img_embeds, atts_img = m.encode_img(pv_slots)
        img_embeds = m.layer_norm(img_embeds).to(self.device)
        wrapped, atts_wrapped = m.prompt_wrap(img_embeds, atts_img.to(self.device))
        wrapped = wrapped.half()

        bsz = wrapped.shape[0]
        bos = torch.full([bsz, 1], m.llama_tokenizer.bos_token_id,
                         dtype=torch.long, device=self.device)
        gen = gen_ids.unsqueeze(0).expand(bsz, -1).to(self.device)

        inputs_embeds = torch.cat([m.embed_tokens(bos).to(wrapped.dtype), wrapped,
                                   m.embed_tokens(gen).to(wrapped.dtype)], dim=1)
        attention_mask = torch.cat([
            torch.ones([bsz, 1], dtype=torch.long, device=self.device),
            atts_wrapped,
            torch.ones(gen.shape, dtype=torch.long, device=self.device)], dim=1)

        out = m.llama_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                            use_cache=False, return_dict=True)  # 不建 KV cache, 省显存

        start = wrapped.shape[1]      # token i 由它前一个位置预测
        logits = out.logits[:, start:start + gen.shape[1], :]
        lp = logits.float().log_softmax(dim=-1).gather(-1, gen.unsqueeze(-1)).squeeze(-1)
        return lp.cpu().numpy()

    def _run(self, pv_slots: List[torch.Tensor], gen_ids: torch.Tensor,
             slot: int, masks: Sequence[Sequence[int]]) -> np.ndarray:
        """
        对 slot 号影像施加每种遮挡, 返回 [len(masks), T]。
        其余影像一并全遮 —— 否则"只留一格"的语义会被另一张完整的图破坏。
        """
        rows, bs = [], max(1, self.batch_size)
        allc = list(range(self.grid * self.grid))
        for i in range(0, len(masks), bs):
            chunk = masks[i:i + bs]
            batched = []
            for si, pv in enumerate(pv_slots):
                if si == slot:
                    batched.append(torch.cat([self._occlude(pv, mk) for mk in chunk], dim=0))
                else:
                    blank = self._occlude(pv, allc)
                    batched.append(blank.expand(len(chunk), -1, -1, -1).contiguous())
            try:
                rows.append(self._logprobs(batched, gen_ids))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.batch_size > 1:
                    print(f"[heatmap] 显存不足, batch {self.batch_size} -> 1")
                    self.batch_size = 1
                for mk in chunk:
                    single = [self._occlude(pv, mk) if si == slot
                              else self._occlude(pv, allc)
                              for si, pv in enumerate(pv_slots)]
                    rows.append(self._logprobs(single, gen_ids))
        return np.concatenate(rows, axis=0)

    def explain(self, images: List[Image.Image], gen_ids: torch.Tensor,
                report_text: str, views: List[str]) -> Dict:
        side = self.grid
        cells = list(range(side * side))
        pv_slots = [self.to_pixel_values(im) for im in images]

        ids = gen_ids.tolist()
        special = set(self.model.llama_tokenizer.all_special_ids)
        valid = [i for i, t in enumerate(ids) if t not in special]

        hits = find_keywords(report_text)
        status, reason = assess(report_text, hits)
        if not valid:
            return {"status": "normal", "status_reason": "没有可用的文本 token",
                    "impression": build_impression("normal", []),
                    "heatmaps": [], "findings": [], "image_contribution": None}

        # 基线与全遮挡对照
        base = self._logprobs([pv.clone() for pv in pv_slots], gen_ids)[0]
        blank_lp = self._logprobs([self._occlude(pv, cells) for pv in pv_slots], gen_ids)[0]
        contribution = float((base - blank_lp)[valid].mean())

        # 只保留一格, 相对全遮挡的恢复量
        masks = [[c for c in cells if c != cell] for cell in cells]
        recover = np.stack([self._run(pv_slots, gen_ids, s, masks) - blank_lp[None, :]
                            for s in range(len(pv_slots))])   # [slots, cells, T]

        def maps(rows: List[int]) -> List[np.ndarray]:
            m = recover[:, :, rows].mean(axis=2)
            return [m[s].reshape(side, side) for s in range(m.shape[0])]

        # token 与字符位置对齐（SentencePiece 会把病名切成多个 subword）
        spans, prev = [], ""
        for i in range(len(ids)):
            cur = self.model.llama_tokenizer.decode(ids[:i + 1], skip_special_tokens=True)
            spans.append((len(prev), len(cur)))
            prev = cur

        overall = maps(valid)

        # 模型对整份报告的平均 token 概率。
        # 注意: 这是语言模型的生成置信度, 不是校准过的诊断概率 ——
        # 前端必须如实标注, 不能当成"该疾病的可能性"展示。
        report_conf = float(np.exp(base[valid].mean()))

        result = {
            "status": status,
            "status_reason": reason,
            "impression": build_impression(status, hits),
            "report_confidence": round(report_conf, 4),
            "image_contribution": round(contribution, 4),
            "heatmaps": [],
            "findings": [],
        }

        # uncertain 态才给整体图; findings 态只给词级图, 避免两张图混淆;
        # normal 态什么都不给
        if status == "uncertain":
            result["heatmaps"] = [
                {"view": views[i], "image": render_overlay(im, overall[i], self.out_size)}
                for i, im in enumerate(images)]

        for hit in hits:
            s, e = hit["span"]
            rows = [i for i in valid if spans[i][1] > s and spans[i][0] < e]
            if not rows:
                continue
            # 减掉整体基线 -> 词特异定位。两张图单位都是 nats、量级一致, 直接相减即可
            grids = [np.maximum(g - b, 0.0) for g, b in zip(maps(rows), overall)]
            result["findings"].append({
                "label": hit["label"],
                "keyword": hit["keyword"],
                "char_span": [s, e],
                # 该病名 token 的平均概率。同样是语言置信度而非诊断概率。
                "confidence": round(float(np.exp(base[rows].mean())), 4),
                "maps": [{"view": views[i],
                          "image": render_overlay(im, grids[i], self.out_size)}
                         for i, im in enumerate(images)],
            })
        return result


# 模型加载与生成
def load_model():
    global model, image_processor, explainer
    print("正在加载模型...")

    args = Namespace(
        vision_model=SWIN_MODEL, llama_model=LLAMA_MODEL,
        freeze_vm=True, vis_use_lora=True, vis_r=16, vis_alpha=16,
        llm_use_lora=False, llm_r=16, llm_alpha=16, lora_dropout=0.1,
        low_resource=True, global_only=False, end_sym='</s>',
        savedmodel_path='save/iu_xray/v1_delta', ckpt_file=None,
        delta_file=CHECKPOINT_PATH, max_length=60, beam_size=3, do_sample=False,
        no_repeat_ngram_size=2, num_beam_groups=1, min_new_tokens=40,
        max_new_tokens=100, repetition_penalty=2.0, length_penalty=2.0,
        diversity_penalty=0, temperature=0, prompt="", img_size=224,
        num_layers=1, num_query_token=32, weights=[0.5, 0.5],
        scorer_types=['Bleu_4', 'CIDEr'], max_epochs=15, learning_rate=1e-4,
    )

    model = R2GenGPT(args)
    model.eval()
    model.visual_encoder.to(DEVICE)
    model.llama_proj.to(DEVICE)
    model.layer_norm.to(DEVICE)
    image_processor = AutoImageProcessor.from_pretrained(SWIN_MODEL)

    if ENABLE_HEATMAP:
        explainer = OcclusionExplainer(model, image_processor, DEVICE, GRID, BATCH)
        print(f"热力图已启用：遮挡法 {GRID}x{GRID} 网格，批 {BATCH}")

    print("模型加载完成！")


@torch.no_grad()
def generate(pv_slots: List[torch.Tensor]) -> Tuple[str, torch.Tensor]:
    img_embeds, atts_img = model.encode_img(pv_slots)
    img_embeds = model.layer_norm(img_embeds).to(DEVICE)
    img_embeds, atts_img = model.prompt_wrap(img_embeds, atts_img.to(DEVICE))
    img_embeds = img_embeds.half()      # 匹配 4-bit 量化 LLaMA 的精度

    bsz = img_embeds.shape[0]
    bos = torch.full([bsz, 1], model.llama_tokenizer.bos_token_id,
                     dtype=torch.long, device=img_embeds.device)
    inputs_embeds = torch.cat([model.embed_tokens(bos), img_embeds], dim=1)
    attention_mask = torch.cat([
        torch.ones([bsz, 1], dtype=torch.long, device=img_embeds.device),
        atts_img], dim=1)

    outputs = model.llama_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        num_beams=model.hparams.beam_size,
        do_sample=model.hparams.do_sample,
        min_new_tokens=model.hparams.min_new_tokens,
        max_new_tokens=model.hparams.max_new_tokens,
        repetition_penalty=model.hparams.repetition_penalty,
        length_penalty=model.hparams.length_penalty,
        temperature=model.hparams.temperature,
    )
    return model.decode(outputs[0]), outputs[0]


# 接口
@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/health")
async def health():
    return {"status": "ok", "model": "R2GenGPT-Delta-Epoch14",
            "heatmap": explainer is not None}


@app.post("/predict")
async def predict(
    files: List[UploadFile] = File(..., description="胸部X光影像，按 正位、侧位 顺序上传"),
    heatmap: bool = True,
):
    """
    接收正位/侧位影像（1~2 张，按顺序视为 frontal / lateral），生成报告和热力图。

    热力图约需 GRID^2 x 影像数 次前向，双图 7x7 约 13 秒。不需要时传 heatmap=false。
    """
    try:
        pil_images, pv_slots = [], []
        for f in files:
            image = Image.open(io.BytesIO(await f.read())).convert("RGB")
            pil_images.append(image)
            array = np.array(image, dtype=np.uint8)
            pv_slots.append(
                image_processor(array, return_tensors="pt").pixel_values.to(DEVICE))

        views = [VIEW_NAMES[i] if i < len(VIEW_NAMES) else f"view{i + 1}"
                 for i in range(len(pil_images))]

        report_text, gen_ids = generate(pv_slots)
        print(f"生成报告: {report_text}")

        # Impression 由 Findings 里抽到的病名推导，保证两段不矛盾。
        # 关掉热力图时也要有，所以在这里先算一次。
        hits = find_keywords(report_text)
        status_code, _ = assess(report_text, hits)

        resp = {
            "report": report_text,         # 兼容旧字段，内容等于 Findings
            "findings_text": report_text,  # Findings 段：模型生成
            "impression": build_impression(status_code, hits),  # 需医师复核
            "impression_source": "derived",
            "views": views,
            "status": "success",
            "heatmap_path": "",          # 兼容旧字段
        }

        if heatmap and explainer is not None:
            try:
                heat = explainer.explain(pil_images, gen_ids, report_text, views)
                resp.update({
                    # normal / findings / uncertain
                    # normal 态下 heatmaps 和 findings 都为空，前端不应渲染热力图区块
                    "finding_status": heat["status"],
                    "status_reason": heat["status_reason"],
                    "impression": heat["impression"],
                    "heatmaps": heat["heatmaps"],
                    "findings": heat["findings"],
                    # 整张影像对报告的总贡献(nats)。偏低说明报告主要来自语言先验,
                    # 此时热力图参考价值有限
                    "image_contribution": heat["image_contribution"],
                })
                print(f"热力图完成: {heat['status']}, "
                      f"{len(heat['findings'])} 个病灶, "
                      f"影像贡献 {heat['image_contribution']}")
            except Exception as he:
                import traceback
                traceback.print_exc()
                print(f"热力图生成失败，仅返回报告: {he}")

        return resp

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"report": "", "status": "error", "message": str(e),
                "heatmap_path": "", "heatmaps": [], "findings": []}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)