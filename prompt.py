# 文件名：local_doc_qa.py
# 依赖安装：pip install zhipuai pypdf sentence-transformers numpy

import os
import re
from typing import List, Tuple

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from zhipuai import ZhipuAI

# ========== 配置 ==========
API_KEY = os.getenv("ZAI_API_KEY")
DOCUMENT_FILES = ["D:\校友会\勤工助学合同.docx"]  # 你的文档路径列表
MODEL_NAME = "glm-4"  # 也可换成 glm-4-flash 等

# 分块参数
CHUNK_SIZE = 400       # 每块目标字符数
CHUNK_OVERLAP = 80     # 相邻块重叠字符数
TOP_K = 3              # 检索返回最相关的 K 个块

# ========== 文档加载 ==========
def load_documents(file_paths: List[str]) -> List[str]:
    """读取 PDF / TXT / DOCX 文档，返回纯文本列表"""
    texts = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"⚠️ 文件不存在，已跳过：{path}")
            continue

        if path.endswith(".pdf"):
            reader = PdfReader(path)
            content = "\n".join(page.extract_text() or "" for page in reader.pages)

        elif path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

        elif path.endswith(".docx"):
            from docx import Document
            doc = Document(path)
            content = "\n".join(p.text for p in doc.paragraphs)

        elif path.endswith(".doc"):
            print(f"⚠️ 旧版 .doc 格式无法直接解析，请先另存为 .docx：{path}")
            continue

        else:
            print(f"⚠️ 不支持的格式，已跳过：{path}")
            continue

        texts.append(content)
    return texts

# ========== 文本分块 ==========
def split_into_chunks(text: str, source: str = "") -> List[dict]:
    """
    按段落和长度双重策略分块，保留重叠以避免语义割裂。
    返回 [{"text": ..., "source": ...}, ...]
    """
    # 先按空行分段
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        # 如果加入当前段落后超长，就切割
        if len(current) + len(para) > CHUNK_SIZE and current:
            chunks.append(current)
            # 保留重叠部分（从 current 尾部取）
            current = current[-CHUNK_OVERLAP:] + para
        else:
            current += "\n" + para if current else para
    if current:
        chunks.append(current)
    return [{"text": c, "source": source} for c in chunks]

# ========== 向量化与检索 ==========
class LocalRetriever:
    def __init__(self, chunks: List[dict]):
        self.chunks = chunks
        # 改为本地路径加载，模型已下载到当前目录
        self.model = SentenceTransformer("./all-MiniLM-L6-v2", local_files_only=True)
        self.embeddings = self.model.encode([c["text"] for c in chunks], normalize_embeddings=True)

    def search(self, query: str, top_k: int = TOP_K) -> List[dict]:
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self.embeddings, q_emb)  # 余弦相似度
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {**self.chunks[i], "score": float(scores[i])}
            for i in top_indices
        ]

# ========== 生成回答 ==========
def generate_answer(client: ZhipuAI, question: str, retrieved: List[dict]) -> str:
    # 构造上下文
    context_parts = []
    for i, item in enumerate(retrieved, 1):
        context_parts.append(f"[片段{i}] 来源：{item['source']}\n{item['text']}")
    context = "\n\n".join(context_parts)

    prompt = f"""你是一个严谨的文档问答助手。请**只根据以下文档片段**回答用户问题。
如果片段中没有足够信息，请直接说“文档中未找到相关信息”，不要编造。

文档片段：
\"\"\"
{context}
\"\"\"

用户问题：{question}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content

# ========== 主流程 ==========
def main():
    # 1. 初始化智谱客户端（SDK 会自动读取环境变量 ZHIPUAI_API_KEY[citation:2]）
    client = ZhipuAI(api_key=API_KEY)

    # 2. 加载文档
    print("正在加载文档...")
    raw_texts = load_documents(DOCUMENT_FILES)

    # 3. 分块
    all_chunks = []
    for path, text in zip(DOCUMENT_FILES, raw_texts):
        chunks = split_into_chunks(text, source=os.path.basename(path))
        all_chunks.extend(chunks)
    print(f"共生成 {len(all_chunks)} 个文本块。")

    # 4. 构建本地检索器
    print("正在生成向量索引（首次运行会下载嵌入模型）...")
    retriever = LocalRetriever(all_chunks)

    # 5. 交互问答循环
    print("\n文档问答助手已就绪。输入问题开始（输入 quit 退出）：\n")
    while True:
        question = input("你的问题：").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        retrieved = retriever.search(question, top_k=TOP_K)
        print(f"（检索到 {len(retrieved)} 个相关片段，最高相似度 {retrieved[0]['score']:.3f}）")

        answer = generate_answer(client, question, retrieved)
        print(f"\n回答：{answer}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()