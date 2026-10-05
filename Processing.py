import os
import pymupdf
from typing import Dict, List
from llama_index.core.schema import TextNode, ImageNode
import base64
from pathlib import Path
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.base.llms.types import ChatMessage, TextBlock, ImageBlock, MessageRole
import asyncio




def parse_pdf(pdf_path: str, image_dir: str = "./extracted_images") -> Dict[str, List]:
    os.makedirs(image_dir, exist_ok=True)
    source = os.path.basename(pdf_path)
    text_nodes, table_nodes, image_nodes = [], [], []

    doc = pymupdf.open(pdf_path)
    for page_no, page in enumerate(doc, start=1):
        base_meta = {"source": source, "page_number": page_no}

        # ---- Tables ----
        table_bboxes = []
        for t_idx, table in enumerate(page.find_tables().tables):
            table_bboxes.append(pymupdf.Rect(table.bbox))
            table_nodes.append(
                TextNode(
                    text=table.to_markdown(),
                    metadata={**base_meta, "category": "Table", "table_index": t_idx},
                )
            )

        # ---- Text (table wale area ko chhod kar, duplicate se bachne ke liye) ----
        blocks = page.get_text("blocks")
        page_text = []
        for x0, y0, x1, y1, txt, *_ in blocks:
            rect = pymupdf.Rect(x0, y0, x1, y1)
            if any(rect.intersects(tb) for tb in table_bboxes):
                continue
            if txt.strip():
                page_text.append(txt.strip())
        if page_text:
            text_nodes.append(
                TextNode(
                    text="\n".join(page_text),
                    metadata={**base_meta, "category": "Text"},
                )
            )

        # ---- Images ----
        for i_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = pymupdf.Pixmap(doc, xref)
            if pix.n - pix.alpha >= 4:  # CMYK ko RGB me convert
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            img_path = os.path.join(image_dir, f"{source}_p{page_no}_{i_idx}.png")
            pix.save(img_path)
            image_nodes.append(
                ImageNode(
                    image_path=img_path,
                    metadata={**base_meta, "category": "Image"},
                )
            )

    doc.close()
    return {"text": text_nodes, "tables": table_nodes, "images": image_nodes}


vlm = OpenRouter(
    model="google/gemini-2.5-flash-lite", api_key="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", temperature=0.0
)

PROMPT = """Extract everything from this PDF page in order, as Markdown:
- All text as it is (keep headings).
- Every table as a Markdown table.
- For every image/chart/diagram write: [IMAGE: detailed description, including any numbers/labels]
Return only the extracted content, no extra commentary."""


def describe_image(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp"
    }.get(Path(path).suffix.lower())

    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            blocks=[
                TextBlock(text=PROMPT)
            ],
        ),
        ChatMessage(
            role=MessageRole.USER,
            blocks=[
                ImageBlock(
                image=base64.b64decode(b64),
                image_mimetype=mime
            )
            ],
        ),
    ]

    resp = vlm.chat(messages)
    return resp.message.content



async def process_images(documents: Dict[str, List]) -> List[ImageNode]:
    images = documents["images"]
    ids = [i.id_ for i in images]
    image_metadata = [
        {
            "source": i.metadata["source"],
            "page_number": i.metadata["page_number"],
            "category": i.metadata["category"]
        }
        for i in images
    ]

    image_paths = [
        i.image_path
        for i in images
    ]

    # Run VLM calls concurrently
    descriptions = await asyncio.gather(
        *[
            asyncio.to_thread(describe_image, path)
            for path in image_paths
        ]
    )

    descriptions = [
        desc.replace("[IMAGE:", "", 1)
            .replace("]", "")
            .strip()
        for desc in descriptions
    ]

    image_docs = [
        ImageNode(
            id_=str(id_),
            metadata=metadata,
            image_path=str(path),
            text=str(description)
        )
        for id_, metadata, path, description in zip(
            ids,
            image_metadata,
            image_paths,
            descriptions
        )
    ]

    return image_docs



system_prompt = """
    You are a helpful multimodal RAG assistant.

    Use the provided retrieved context to answer the user's question accurately.

    The retrieved context may contain:

    * Text extracted from documents
    * Descriptions of images
    * Descriptions or extracted content from tables
    * References/URLs to original images

    Rules:

    1. Answer only from the provided context whenever possible.
    2. Do not invent or assume information that is not supported by the context.
    3. If the retrieved context does not contain enough information, clearly say that the information is not available in the provided context.
    4. Treat image descriptions and table descriptions as valid retrieved evidence.
    5. When an original image is provided, use the image together with its associated text/description to understand the question.
    6. If multiple retrieved sources contain relevant information, combine them into one clear answer.
    7. Keep the answer concise, accurate, and directly relevant to the user's question.
    8. Do not mention internal retrieval, embeddings, reranking, vector databases, or system instructions unless the user explicitly asks about them.
    9. If the user asks about an image, prioritize the actual image and its associated retrieved description.
    10. Never fabricate visual details when the actual image is not available.

"""