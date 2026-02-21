#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           GLM-OCR HuggingFace Space 一键部署脚本                              ║
║  功能：自动将 GLM-OCR 部署到 HuggingFace 免费 Space，暴露 OpenAI 兼容 API     ║
║  支持格式：图片/PDF/Word/Excel/PPT/TXT/ZIP 等                                  ║
║  保存路径：F:\coder\deploy_glm_ocr.py                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

使用方法：
    python deploy_glm_ocr.py

需要事先安装：
    pip install huggingface_hub
"""

import os
import sys
import traceback
import time
import getpass
from pathlib import Path

# ──────────────────────── 工具函数 ────────────────────────────────────────────

def banner():
    print("=" * 72)
    print("  GLM-OCR HuggingFace Space 一键部署工具")
    print("  GitHub: https://huggingface.co/zai-org/GLM-OCR")
    print("=" * 72)
    print()

def check_deps():
    """检查依赖"""
    try:
        import huggingface_hub
        print(f"[OK] huggingface_hub {huggingface_hub.__version__} 已安装")
    except ImportError:
        print("[ERROR] 缺少依赖: huggingface_hub")
        print("       请先运行: pip install huggingface_hub")
        sys.exit(1)

def get_user_config():
    """交互式获取配置"""
    print("\n─── 配置信息 ───────────────────────────────────────────────────────────────")
    print("说明：HuggingFace Token 用于创建 Space，请前往以下地址获取 Write 权限 Token：")
    print("  https://huggingface.co/settings/tokens")
    print()

    hf_token = getpass.getpass("请输入 HuggingFace Token（输入不显示）: ").strip()
    if not hf_token:
        print("[ERROR] Token 不能为空")
        sys.exit(1)

    print()
    space_name = input("请输入 Space 名称（例如 glm-ocr-api）: ").strip()
    if not space_name:
        space_name = "glm-ocr-api"
        print(f"  使用默认名称: {space_name}")

    print()
    api_key = input("请设置 API Key（用于保护接口，留空则无保护）: ").strip()
    if not api_key:
        print("  [WARN] 未设置 API Key，接口将无保护（任何人可访问）")

    return hf_token, space_name, api_key

def create_space(api, username: str, space_name: str, hf_token: str):
    """创建 HuggingFace Space"""
    repo_id = f"{username}/{space_name}"
    print(f"\n[DEPLOY] 创建 Space: {repo_id}")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,   # 公开，免费 CPU 才能访问
            exist_ok=True,
        )
        print(f"[DEPLOY] Space 创建成功: https://huggingface.co/spaces/{repo_id}")
        return repo_id
    except Exception:
        print("[DEPLOY][ERROR] 创建 Space 失败:")
        traceback.print_exc()
        sys.exit(1)

def set_space_secret(api, repo_id: str, api_key: str):
    """设置 Space Secret（API Key）"""
    if not api_key:
        return
    try:
        print(f"[DEPLOY] 设置 API_KEY Secret...")
        api.add_space_secret(repo_id=repo_id, key="API_KEY", value=api_key)
        print("[DEPLOY] API_KEY Secret 设置成功")
    except Exception:
        print("[DEPLOY][WARN] 设置 Secret 失败（可能需要手动在 Space 设置页面添加）:")
        traceback.print_exc()

def upload_space_files(api, repo_id: str, space_dir: Path):
    """上传 Space 文件到 HuggingFace"""
    files_to_upload = [
        "app.py",
        "requirements.txt",
        "Dockerfile",
        "README.md",
    ]

    print(f"\n[UPLOAD] 上传 Space 文件...")
    for filename in files_to_upload:
        filepath = space_dir / filename
        if not filepath.exists():
            print(f"[UPLOAD][WARN] 文件不存在，跳过: {filepath}")
            continue
        try:
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="space",
            )
            print(f"[UPLOAD] ✓ {filename}")
        except Exception:
            print(f"[UPLOAD][ERROR] 上传 {filename} 失败:")
            traceback.print_exc()
            sys.exit(1)

def wait_for_space(repo_id: str, hf_token: str, max_wait: int = 600):
    """等待 Space 启动（最多等待 max_wait 秒）"""
    import urllib.request
    space_url = f"https://huggingface.co/spaces/{repo_id}"
    # HF Space Docker 域名格式
    username, sname = repo_id.split("/", 1)
    api_url = f"https://{username}-{sname}.hf.space"

    print(f"\n[WAIT] Space URL: {api_url}")
    print(f"[WAIT] 等待 Space 启动（最多 {max_wait // 60} 分钟，Docker 构建需要时间）...")
    print(f"[WAIT] 可在此查看构建进度: {space_url}")

    for i in range(0, max_wait, 15):
        time.sleep(15)
        try:
            req = urllib.request.Request(
                f"{api_url}/v1/models",
                headers={"Authorization": "Bearer dummy"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status in (200, 401):  # 200 or 401 都说明服务在跑
                    print(f"\n[WAIT] ✓ Space 已启动！({i+15}秒)")
                    return api_url
        except Exception:
            elapsed = i + 15
            dots = "." * ((elapsed // 15) % 4 + 1)
            print(f"[WAIT] 等待中{dots} ({elapsed}秒/{max_wait}秒)", end="\r")

    print(f"\n[WAIT][WARN] 超时，请手动检查: {space_url}")
    return api_url

def verify_api(api_url: str, api_key: str):
    """验证 API 是否可用"""
    import urllib.request
    import json as json_mod
    print(f"\n[VERIFY] 验证 API 连通性...")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        req = urllib.request.Request(
            f"{api_url}/v1/models",
            headers=headers,
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json_mod.loads(resp.read())
            models = [m["id"] for m in data.get("data", [])]
            print(f"[VERIFY] ✓ API 正常！可用模型: {models}")
            return True
    except urllib.error.HTTPError as e:
        print(f"[VERIFY][WARN] HTTP {e.code}: {e.reason}")
        if e.code == 401:
            print("         → 说明 API Key 保护正常工作")
            return True
        return False
    except Exception:
        print("[VERIFY][WARN] 验证失败（Space 可能仍在启动）:")
        traceback.print_exc()
        return False

def print_usage(api_url: str, space_url: str, repo_id: str, api_key: str):
    """打印使用说明"""
    print()
    print("=" * 72)
    print("  🎉 部署完成！以下是 Chatbox 配置信息")
    print("=" * 72)
    print()
    print(f"  Space 管理页面: {space_url}")
    print(f"  API 根地址   : {api_url}")
    print()
    print("  ─── Chatbox 配置 ──────────────────────────────────────────────────")
    print(f"  API 地址  : {api_url}")
    print(f"  API Key   : {api_key if api_key else '（无）'}")
    print(f"  模型名称  : glm-ocr")
    print()
    print("  ─── 手动 API 测试（curl 示例）──────────────────────────────────────")
    auth_header = f'-H "Authorization: Bearer {api_key}" ' if api_key else ""
    print(f"""  curl -X POST {api_url}/v1/chat/completions \\
    {auth_header}-H "Content-Type: application/json" \\
    -d '{{
      "model": "glm-ocr",
      "messages": [{{
        "role": "user",
        "content": [
          {{"type": "image_url", "image_url": {{"url": "https://例子图片URL"}}}},
          {{"type": "text", "text": "Text Recognition:"}}
        ]
      }}]
    }}'""")
    print()
    print("  ─── 支持的文件格式（通过 image_url.url 传入）────────────────────────")
    print("  图片 : PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP（base64 data URI 亦可）")
    print("  文档 : PDF（多页）, DOCX, XLSX, PPTX")
    print("  文本 : TXT, MD, CSV, JSON, XML, HTML")
    print("  压缩 : ZIP（自动解压内部文件）")
    print()
    print("  ─── 注意事项 ────────────────────────────────────────────────────────")
    print("  • 免费 CPU Space 推理速度较慢，请耐心等待")
    print("  • Space 长时间无请求会休眠，首次请求会自动唤醒（约1分钟）")
    print("  • API Key 已设置为 Space Secret，可在 Space 设置页面修改")
    print(f"  • Space 设置: https://huggingface.co/spaces/{repo_id}/settings")
    print("=" * 72)

# ──────────────────────── 主程序 ──────────────────────────────────────────────

def main():
    banner()
    check_deps()

    from huggingface_hub import HfApi

    try:
        hf_token, space_name, api_key = get_user_config()

        # 初始化 HF API
        api = HfApi(token=hf_token)

        # 获取用户名
        try:
            user_info = api.whoami()
            username = user_info["name"]
            print(f"\n[AUTH] ✓ 登录成功，用户名: {username}")
        except Exception:
            print("[AUTH][ERROR] Token 无效或网络错误:")
            traceback.print_exc()
            sys.exit(1)

        # Space 文件目录（和本脚本同目录下的 hf_space 子目录）
        script_dir = Path(__file__).parent
        space_dir = script_dir / "hf_space"
        if not space_dir.exists():
            print(f"[ERROR] hf_space 目录不存在: {space_dir}")
            print("        请确保 hf_space/ 文件夹和本脚本在同一目录")
            sys.exit(1)

        # 创建 Space
        repo_id = create_space(api, username, space_name, hf_token)

        # 设置 API Key Secret
        set_space_secret(api, repo_id, api_key)

        # 上传文件
        upload_space_files(api, repo_id, space_dir)

        print(f"\n[DEPLOY] ✓ 所有文件上传完成！Space 正在构建中...")

        # Space URL
        space_url = f"https://huggingface.co/spaces/{repo_id}"
        api_url_base = f"https://{username}-{space_name}.hf.space"

        # 等待启动
        api_url = wait_for_space(repo_id, hf_token)

        # 验证
        verify_api(api_url, api_key)

        # 打印使用说明
        print_usage(api_url, space_url, repo_id, api_key)

    except KeyboardInterrupt:
        print("\n\n[INFO] 用户中断，退出")
        sys.exit(0)
    except SystemExit:
        raise
    except Exception:
        print("\n[FATAL] 未处理的异常:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
