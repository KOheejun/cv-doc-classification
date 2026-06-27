import os, glob, json, argparse
import numpy as np
import pandas as pd

def normalize_id(x):
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8")
    x = str(x)
    x = os.path.splitext(x)[0]
    return x

def find_image(img_dir, img_id):
    img_id = normalize_id(img_id)
    pats = [
        os.path.join(img_dir, f"{img_id}.jpg"),
        os.path.join(img_dir, f"{img_id}.png"),
        os.path.join(img_dir, f"{img_id}.jpeg"),
        os.path.join(img_dir, f"{img_id}.*"),
        os.path.join(img_dir, "**", f"{img_id}.*"),
    ]
    for p in pats:
        hits = glob.glob(p, recursive=True)
        if hits:
            return hits[0]
    return None

def crop_region(img, mode):
    h, w = img.shape[:2]
    if mode == "title":
        return img[:int(h * 0.28), :]
    if mode == "body":
        return img[int(h * 0.28):int(h * 0.85), :]
    return img

def contains_any(text, kws):
    return any(k in text for k in kws)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["closecall_all37", "bias7_cond"], required=True)

    ap.add_argument("--conv_w", type=float, default=0.69)
    ap.add_argument("--swin_w", type=float, default=0.31)

    ap.add_argument("--bias_eps", type=float, default=0.02)
    ap.add_argument("--bias_margin", type=float, default=0.20, help="abs(p3-p7) < this => bias allowed")
    ap.add_argument("--bias_sum", type=float, default=0.70, help="p3+p7 > this => bias allowed")

    ap.add_argument("--test_img_dir", default="data/test")
    ap.add_argument("--cache", default="oof/ocr_cache_test_3_7_v2.json")
    ap.add_argument("--gpu", type=int, default=1)

    args = ap.parse_args()

    # 3/7 키워드 (너가 찾은 핵심 + 외래 보강)
    KW3 = ["입원", "퇴원", "입퇴원"]
    KW7 = ["통원", "외래"]

    ids = np.load("oof/convnext_small_test_ids.npy", allow_pickle=True)
    Pc  = np.load("oof/convnext_small_test_probs.npy")
    Ps  = np.load("oof/swin_small_test_probs.npy")

    P = args.conv_w * Pc + args.swin_w * Ps
    pred_base = P.argmax(1)
    pred_new  = pred_base.copy()

    try:
        with open(args.cache, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except:
        cache = {}

    import cv2
    import easyocr
    reader = easyocr.Reader(['ko'], gpu=bool(args.gpu))

    n_candidates = 0
    ocr_used = 0
    changed = 0
    missing_img = 0

    for i, img_id in enumerate(ids):
        # 개선 closecall: "예측이 3 또는 7"인 샘플 전부 OCR 대상
        if pred_base[i] not in (3, 7):
            continue

        n_candidates += 1

        key = str(img_id)
        if key in cache:
            title_text = cache[key].get("title", "")
            body_text  = cache[key].get("body", "")
        else:
            path = find_image(args.test_img_dir, img_id)
            if path is None:
                missing_img += 1
                continue
            img = cv2.imread(path)
            if img is None:
                missing_img += 1
                continue

            title = crop_region(img, "title")
            body  = crop_region(img, "body")

            title_text = " ".join(reader.readtext(title, detail=0, paragraph=True))
            body_text  = " ".join(reader.readtext(body,  detail=0, paragraph=True))
            cache[key] = {"title": title_text, "body": body_text}

        ocr_used += 1

        title_has_3 = contains_any(title_text, KW3)
        title_has_7 = contains_any(title_text, KW7)
        body_has_3  = contains_any(body_text, KW3)
        body_has_7  = contains_any(body_text, KW7)

        decided = None

        # 혼합 제목(입원+통원 같이 쓰는 경우): 제목 무시, 본문으로만 판단
        if title_has_3 and title_has_7:
            if body_has_3 and not body_has_7:
                decided = 3
            elif body_has_7 and not body_has_3:
                decided = 7
        else:
            # 제목/본문 어디든 한쪽만 확실하면 결정
            has3 = (title_has_3 or body_has_3)
            has7 = (title_has_7 or body_has_7)
            if has3 and not has7:
                decided = 3
            elif has7 and not has3:
                decided = 7

        old = int(pred_new[i])

        if decided is not None:
            pred_new[i] = decided
        else:
            # 개선 bias7: 키워드가 둘 다 없을 때, "3↔7 박빙"이면 아주 약하게 7 쪽으로만
            if args.mode == "bias7_cond":
                p3 = float(P[i, 3]); p7 = float(P[i, 7])
                if (abs(p3 - p7) < args.bias_margin) and ((p3 + p7) > args.bias_sum):
                    P2 = P[i].copy()
                    P2[7] += args.bias_eps
                    pred_new[i] = int(P2.argmax())

        if int(pred_new[i]) != old:
            changed += 1

    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    with open(args.cache, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    os.makedirs("submissions", exist_ok=True)

    if args.mode == "closecall_all37":
        out = f"submissions/submission_ocr_closecall_all37_base069031.csv"
    else:
        out = f"submissions/submission_ocr_bias7cond_base069031_eps{args.bias_eps:.2f}_m{args.bias_margin:.2f}_s{args.bias_sum:.2f}.csv"

    pd.DataFrame({"ID": ids, "target": pred_new}).to_csv(out, index=False)

    print(f"[DONE] {out}")
    print(f"[STATS] samples={len(ids)} candidates(3or7)={n_candidates} ocr_used={ocr_used} missing_img={missing_img} changed={changed}")
    print(f"[MODE] {args.mode} eps={args.bias_eps} bias_margin={args.bias_margin} bias_sum={args.bias_sum} gpu={args.gpu}")

if __name__ == "__main__":
    main()
