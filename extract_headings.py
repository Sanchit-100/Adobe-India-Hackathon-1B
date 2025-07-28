import fitz  # PyMuPDF
import joblib
import json
import pandas as pd
import re
import argparse
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR   = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
INPUT_DIR    = SCRIPT_DIR / "input"    # Changed from Path("/app/input")
OUTPUT_DIR   = SCRIPT_DIR / "output"
MODEL_DIR    = SCRIPT_DIR / "model"
MODEL_PATH   = MODEL_DIR / "lgbm_binary_classifier.pkl"

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Regex patterns for noise filtering
date_pattern     = re.compile(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}", re.IGNORECASE)
footer_pattern   = re.compile(r"(?i)page\s*\d+|^\d+\s+\d+\s+\d+\s+\d+$|version|copyright")
repeated_numbers = re.compile(r"^\d+(?:\s+\d+)+$")

# Bullet detection
def is_bullet_point(text: str) -> bool:
    return bool(re.match(r'^[•\-\*\+◦◆◇○●■□▪▫]\s+', text))

# Font weight helper
def get_font_weight(font_name: str) -> int:
    nl = font_name.lower() if font_name else ''
    if 'black' in nl or 'heavy' in nl: return 2
    if 'bold' in nl: return 1
    return 0

# PDF parsing & merging lines (copy of infer4 version)
def parse_pdf_and_merge_lines_v2(pdf_path: Path):
    doc = fitz.open(str(pdf_path))
    all_blocks = []
    for page_idx, page in enumerate(doc):
        w, h = page.rect.width, page.rect.height
        lines = []
        for block in page.get_text('dict', flags=11)['blocks']:
            if block['type'] != 0: continue
            for line in block['lines']:
                txt = ' '.join(span['text'].strip() for span in line['spans'] if span['text'].strip())
                if not txt: continue
                avg_size = sum(s['size'] for s in line['spans']) / len(line['spans'])
                font_name = max(set(s['font'] for s in line['spans']), key=lambda f: sum(1 for s in line['spans'] if s['font']==f))
                lines.append({
                    'text': txt, 'bbox': line['bbox'], 'size': avg_size,
                    'font': font_name, 'page_num': page_idx+1,
                    'page_width': w, 'page_height': h
                })
        if not lines: continue
        lines.sort(key=lambda L: L['bbox'][1])
        grouped, cur = [], [lines[0]]
        for ln in lines[1:]:
            prev = cur[-1]
            gap = ln['bbox'][1] - prev['bbox'][3]
            same = abs(ln['size']-prev['size'])<1 and get_font_weight(ln['font'])==get_font_weight(prev['font'])
            if gap < prev['size']*0.5 and same:
                cur.append(ln)
            else:
                grouped.append(cur)
                cur = [ln]
        grouped.append(cur)
        for grp in grouped:
            xs = [L['bbox'][0] for L in grp]; ys = [L['bbox'][1] for L in grp]
            x2 = [L['bbox'][2] for L in grp]; y2 = [L['bbox'][3] for L in grp]
            all_blocks.append({
                'text': ' '.join(L['text'] for L in grp),
                'bbox': (min(xs),min(ys),max(x2),max(y2)),
                'font_size': grp[0]['size'], 'font_name': grp[0]['font'],
                'page_num': grp[0]['page_num'], 'page_width': w, 'page_height': h
            })
    doc.close()
    return all_blocks

# Feature engineering (copy of infer4 version)
def engineer_features(blocks):
    df = pd.DataFrame(blocks)
    if df.empty: return df
    df['text']      = df['text'].astype(str)
    df['font_name'] = df['font_name'].astype(str)
    # body font size (mode of longer lines)
    long = df[df['text'].str.split().str.len()>6]
    body_size = long['font_size'].mode().iat[0] if not long.empty else 10.0
    df['font_weight']       = df['font_name'].apply(get_font_weight)
    df['x']                 = df['bbox'].apply(lambda b: b[0])
    df['y']                 = df['bbox'].apply(lambda b: b[1])
    df['w']                 = df['bbox'].apply(lambda b: b[2]-b[0])
    df['norm_x']            = df['x']/df['page_width']
    df['norm_y']            = df['y']/df['page_height']
    df['text_length']       = df['text'].str.len()
    df['num_words']         = df['text'].str.split().str.len()
    df['is_all_caps']       = df['text'].apply(lambda t:1 if t.isupper() and len(t.strip())>1 else 0)
    cx = df['x'] + df['w']/2
    df['is_centered']       = (abs(cx - df['page_width']/2) < df['page_width']*0.15).astype(int)
    df['numbering_depth']   = df['text'].str.strip().apply(lambda t:len(re.match(r'^(\d+(?:\.\d+)*)',t).group(1).split('.')) if re.match(r'^(\d+(?:\.\d+)*)',t) else 0)
    df['relative_font_size']= df['font_size']/body_size
    df['percent_punct']     = df['text'].apply(lambda t: sum(1 for c in t if c in r'''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~''')/len(t) if t else 0)
    return df

# Prediction and rule-based filtering
FEATURE_COLS = [
    'font_size','font_weight','norm_x','norm_y',
    'text_length','num_words','is_all_caps','is_centered',
    'numbering_depth','relative_font_size','percent_punct'
]

def infer_pdf_binary(pdf_path: Path):
    blocks = parse_pdf_and_merge_lines_v2(pdf_path)
    feats = engineer_features(blocks)
    if feats.empty: return {'outline': []}
    # Identify running header/footer texts (blocks in top 10% or bottom 10%)
    hf = pd.concat([
        feats[feats['norm_y'] < 0.1],
        feats[feats['norm_y'] > 0.9]
    ])
    drop_texts = (
        hf.groupby('text')['page_num']
          .nunique()
          .loc[lambda s: s > 3]
          .index
    )
    X = feats[FEATURE_COLS]
    if X.empty: return {'outline': []}
    preds = model.predict(X)
    feats['is_heading'] = preds
    hdrs = feats[feats['is_heading']==1].copy()
    # Remove any candidate matching running headers/footers
    hdrs = hdrs[~hdrs['text'].isin(drop_texts)]
    # rule-based false-positive removal
    hdrs = hdrs[hdrs['percent_punct']<0.4]
    hdrs = hdrs[~hdrs['text'].str.match(date_pattern, na=False)]
    hdrs = hdrs[~hdrs['text'].str.match(footer_pattern, na=False)]
    hdrs = hdrs[~hdrs['text'].str.match(repeated_numbers, na=False)]
    hdrs = hdrs[~hdrs['text'].apply(is_bullet_point)]
    hdrs = hdrs[~hdrs['text'].str.fullmatch(r'[A-Za-z]{1,2}', na=False)]
    # Remove table-like rows: only if both multiple columns and many numbers
    tbl_pattern = hdrs['text'].str.contains(r"\S+(?: {2,}\S+)+", regex=True)
    num_pattern = hdrs['text'].str.count(r'\d+') > 2
    hdrs = hdrs[~(tbl_pattern & num_pattern)]
    # Remove overly long headings (likely false positives)
    hdrs = hdrs[hdrs['num_words'] <= 15]
    # Sort by location
    final = hdrs.sort_values(['page_num','norm_y'])
    if final.empty:
        return {'title': '', 'outline': []}

    # Assign heading levels hierarchically based on relative font sizes
    tol = 0.05  # relative_font_size tolerance for same level
    levels = []
    stack = []  # stores tuples of (level, relative_font_size)
    for _, row in final.iterrows():
        cur_fs = row['relative_font_size']
        if not stack:
            lvl = 1
            stack.append((lvl, cur_fs))
        else:
            prev_lvl, prev_fs = stack[-1]
            if cur_fs < prev_fs - tol:
                lvl = min(prev_lvl + 1, 6)
                stack.append((lvl, cur_fs))
            elif abs(cur_fs - prev_fs) <= tol:
                lvl = prev_lvl
                stack[-1] = (lvl, cur_fs)
            else:
                # ascend to find matching or parent level
                for i in range(len(stack) - 1, -1, -1):
                    par_lvl, par_fs = stack[i]
                    if abs(cur_fs - par_fs) <= tol:
                        lvl = par_lvl
                        stack = stack[:i+1]
                        stack[-1] = (lvl, cur_fs)
                        break
                    elif cur_fs < par_fs - tol:
                        lvl = min(par_lvl + 1, 6)
                        stack = stack[:i+1]
                        stack.append((lvl, cur_fs))
                        break
                else:
                    lvl = 1
                    stack = [(1, cur_fs)]
        levels.append(f"H{lvl}")
    final['level'] = levels

    # Title detection: top-level header on page 1
    title = ''
    p1 = final[final['page_num'] == 1]
    if not p1.empty:
        h1 = p1[p1['level'] == 'H1']
        candidates = h1 if not h1.empty else p1
        candidates = candidates.copy()
        candidates['score'] = candidates['relative_font_size'] - candidates['norm_y']
        best = candidates.loc[candidates['score'].idxmax()]
        title = best['text'].strip()

    # Build outline
    outline = [
        {'level': r['level'], 'text': r['text'].strip(), 'page': int(r['page_num'])}
        for _, r in final.iterrows()
    ]
    return {'title': title, 'outline': outline}

def extract_all_headings(input_dir, output_dir, model_path=None):
    """Extract headings JSONs for all PDFs in input_dir into output_dir using the trained model."""
    global model
    if model_path:
        model = joblib.load(model_path)
    else:
        model = joblib.load(MODEL_PATH)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for pdf in input_dir.glob('*.pdf'):
        print(f"▶️ Processing {pdf.name}")
        res = infer_pdf_binary(pdf)
        out_file = output_dir / f"{pdf.stem}.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Wrote {out_file.name} ({len(res.get('outline', []))} headings)")

# Replace old main with CLI wrapper
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Extract headings from PDFs")
    parser.add_argument('input_dir', help='Directory containing PDFs')
    parser.add_argument('output_dir', help='Directory to write JSON headings')
    parser.add_argument('--model', help='Path to classifier model', default=None)
    args = parser.parse_args()
    extract_all_headings(args.input_dir, args.output_dir, args.model)