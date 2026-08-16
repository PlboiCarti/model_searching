import json
import logging
import base64
import os
from pathlib import Path

import faiss
import numpy as np

from backend.config import FAISS_INDEX_PATH, FAISS_METADATA_PATH, resolve_path
from backend.embedding.clip_encoder import encode_text_raw, translate_vi_to_en
from backend.embedding.search_algorithms import mmr_search, temporal_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_base64_image(image_path_str):
    try:
        path = resolve_path(image_path_str)
        if path.exists():
            with open(path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        logger.warning(f"Could not load image {image_path_str}: {e}")
    return None

def run_comprehensive_test():
    if not FAISS_INDEX_PATH.exists() or not FAISS_METADATA_PATH.exists():
        logger.error("FAISS index or metadata not found. Please run pipeline steps 1-3 first.")
        return

    logger.info(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    
    logger.info(f"Loading Metadata from {FAISS_METADATA_PATH}...")
    with open(FAISS_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    test_queries = [
        "a photo of a person wearing a red shirt",
        "car moving on the highway",
        "một nhóm người đang đi bộ trên đường",
        "bản tin thời sự -> buổi sáng" # Temporal search test
    ]

    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Comprehensive Model Test Report</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }
            h1 { text-align: center; color: #4CAF50; border-bottom: 2px solid #333; padding-bottom: 10px; }
            .query-section { margin-bottom: 40px; background-color: #1e1e1e; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
            h2 { color: #2196F3; }
            h3 { color: #FF9800; font-size: 1.1em; }
            .results-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; margin-top: 15px; }
            .result-card { background-color: #2a2a2a; border-radius: 8px; padding: 10px; text-align: center; transition: transform 0.2s; border: 1px solid #444; }
            .result-card:hover { transform: scale(1.05); border-color: #4CAF50; }
            .result-card img { max-width: 100%; border-radius: 4px; height: auto; max-height: 200px; object-fit: cover; }
            .result-details { margin-top: 10px; font-size: 0.9em; line-height: 1.4; }
            .score { font-weight: bold; color: #E91E63; }
            .info { color: #B0BEC5; margin: 5px 0; }
        </style>
    </head>
    <body>
        <h1>🚀 CLIP LoRA Fine-tune - Comprehensive Test Report</h1>
    """

    for query in test_queries:
        logger.info(f"Testing query: '{query}'")
        
        is_temporal = "->" in query
        top_k = 5
        
        def encode_fn(text):
            text_en = translate_vi_to_en(text)
            query_vec = encode_text_raw(text_en)
            query_super = query_vec.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_super)
            return query_super

        html_content += f'<div class="query-section">'
        html_content += f'<h2>🔍 Query: "{query}"</h2>'

        if is_temporal:
            scores, results, sub_queries = temporal_search(
                temporal_query_text=query,
                encode_fn=encode_fn,
                index=index,
                metadata=metadata,
                max_gap_sec=120.0,
                top_k_candidates=50,
                top_k=top_k
            )
            html_content += f'<h3>Sub-Queries (Temporal): {" -> ".join(sub_queries)}</h3>'
        else:
            query_en = translate_vi_to_en(query)
            query_super = encode_fn(query_en)
            html_content += f'<h3>Translated (MMR Search): "{query_en}"</h3>'
            scores, results = mmr_search(query_super, index, metadata, top_k=top_k, lambda_mult=0.5, fetch_k=50)

        html_content += '<div class="results-grid">'
        for i, (score, p) in enumerate(zip(scores, results)):
            vid = p.get('video_id', 'N/A')
            fid = p.get('frame_id', 0)
            pts = p.get('pts_time', 0.0)
            path_str = p.get('path', '')
            
            b64_img = get_base64_image(path_str)
            img_tag = f'<img src="data:image/jpeg;base64,{b64_img}" alt="{vid}_{fid}">' if b64_img else '<p style="color: red;">Image missing</p>'
            
            html_content += f"""
            <div class="result-card">
                {img_tag}
                <div class="result-details">
                    <p class="info" style="font-size: 1.1em; color: white;">Rank: #{i+1} | Score: <span class="score">{score:.4f}</span></p>
                    <p class="info">Video: <b>{vid}</b> | Frame: <b>{fid}</b></p>
                    <p class="info">Time: <b>{pts:.1f}s</b></p>
            """
            if is_temporal:
                gap = p.get('time_gap', 0.0)
                pts_a = p.get('event_a_pts', 0.0)
                html_content += f'<p class="info" style="color: #4CAF50;">Gap: {gap:.1f}s (from {pts_a:.1f}s)</p>'
                
            html_content += """
                </div>
            </div>
            """
        html_content += '</div></div>'

    html_content += """
    </body>
    </html>
    """

    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"===========================================================")
    logger.info(f"Report generated successfully: {report_path}")
    logger.info(f"Mở file trên trình duyệt để xem kết quả trực quan.")
    logger.info(f"===========================================================")

if __name__ == "__main__":
    run_comprehensive_test()
