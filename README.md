# GLM-OCR · OpenAI Compatible API on HuggingFace Space

[![Sync to HF Space](https://github.com/YOUR_GITHUB_USERNAME/glm-ocr-api/actions/workflows/sync_to_hf.yml/badge.svg)](https://github.com/YOUR_GITHUB_USERNAME/glm-ocr-api/actions)
[![Model: zai-org/GLM-OCR](https://img.shields.io/badge/Model-zai--org%2FGLM--OCR-blue)](https://huggingface.co/zai-org/GLM-OCR)

将 [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR)（0.9B 多模态 OCR 模型）部署到 HuggingFace 免费 Space，暴露 **OpenAI 兼容 API**，可直接在 Chatbox、ChatGPT Next Web 等客户端接入。

---

## 📁 项目结构

```
.
├── deploy_glm_ocr.py          # 本地一键部署脚本（推荐首次使用）
├── hf_space/                  # 上传到 HuggingFace Space 的所有文件
│   ├── app.py                 # FastAPI OpenAI 兼容 API 服务
│   ├── requirements.txt       # Python 依赖
│   ├── Dockerfile             # Docker 构建
│   └── README.md              # Space 说明
└── .github/workflows/
    └── sync_to_hf.yml         # 推送 main 分支时自动同步到 HF Space
```

---

## 🚀 部署方式

### 方式一：本地一键脚本（推荐）

```bash
pip install huggingface_hub
python deploy_glm_ocr.py
```

按提示输入 HuggingFace Token、Space 名称和 API Key，脚本全自动完成。

### 方式二：GitHub → 自动同步到 HF Space

**配置 GitHub Secrets（仅需一次）：**

在 GitHub 仓库 → **Settings → Secrets and variables → Actions** 添加：

| Secret 名称 | 值 |
|-------------|---|
| `HF_TOKEN` | HuggingFace Write Token（https://huggingface.co/settings/tokens） |
| `HF_SPACE_ID` | 你的 Space ID，格式：`用户名/space名称`（如 `alice/glm-ocr-api`） |

之后每次 push `hf_space/` 目录下的文件，GitHub Actions 自动同步到 HF Space。

---

## 🔌 Chatbox / OpenAI 客户端配置

| 配置项 | 值 |
|--------|---|
| **API 地址** | `https://你的HF用户名-你的Space名称.hf.space` |
| **API Key** | 部署时设置的密钥 |
| **模型名称** | `glm-ocr` |

---

## 📄 支持的文件格式

| 类别 | 格式 |
|------|------|
| 图片 | PNG · JPG · JPEG · GIF · BMP · TIFF · WEBP · base64 |
| 文档 | PDF（多页）· DOCX · XLSX · PPTX |
| 文本 | TXT · MD · CSV · JSON · XML · HTML |
| 压缩 | ZIP（递归解压） |

---

## 📡 API 示例

```bash
curl -X POST https://你的用户名-glm-ocr-api.hf.space/v1/chat/completions \
  -H "Authorization: Bearer 你的API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-ocr",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": "Text Recognition:"}
      ]
    }]
  }'
```

---

## ⚠️ 注意事项

- 免费 CPU Space 推理较慢（约 30–120 秒/图）
- Space 空闲后会自动休眠，首次请求约需 1 分钟唤醒
- 修改 API Key：前往 Space 设置页面 → Secrets → 修改 `API_KEY`

---

## License

MIT
