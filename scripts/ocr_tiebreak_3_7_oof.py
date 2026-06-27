import os, glob, json, argparse
import numpy as np
from sklearn.metrics import f1_score

def find_image(img_dir, img_id):
    img_id = normalize_id(img_id)
    pats = [
        os.path.join(img_dir, f"{img_id}.jpg"),
        os.path.join(img_dir, f"{img_id}.png"),
        os.path.join(img_dir, f"{img_id}.jpeg"),
        os.path.join(img_dir, f"{img_id}.*"),
        os.path.join(img_dir, "**", f"{img_id}.*"),  # 하위폴더까지 재귀 탐색
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


def normalize_id(x):
    # bytes -> str, 그리고 확장자 제거
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8")
    x = str(x)
    x = os.path.splitext(x)[0]
    return x

def contains_any(text, keywords):
    return any(k in text for k in keywords)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["closecall", "bias7"], default="closecall")
    ap.add_argument("--conv_w", type=float, default=0.69)
    ap.add_argument("--swin_w", type=float, default=0.31)
    ap.add_argument("--close_margin", type=float, default=0.12, help="|p3-p7| < margin")
    ap.add_argument("--close_sum", type=float, default=0.60, help="p3+p7 > sum")
    ap.add_argument("--bias_eps", type=float, default=0.02)
    ap.add_argument("--bias_margin", type=float, default=0.20)
    ap.add_argument("--bias_sum", type=float, default=0.70)
    ap.add_argument("--train_img_dir", default="data/train")
    ap.add_argument("--cache", default="oof/ocr_cache_3_7_oof.json")
    ap.add_argument("--gpu", type=int, default=1, help="1=gpu, 0=cpu for easyocr")
    args = ap.parse_args()

    # 키워드 (너가 찾은 핵심 신호 + 외래 보강)
    KW3 = ["입원", "퇴원", "입퇴원"]
    KW7 = ["통원", "외래"]

    # 캐시 로드(속도 개선)
    try:
        with open(args.cache, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except:
        cache = {}

    import cv2
    import easyocr
    reader = easyocr.Reader(['ko'], gpu=bool(args.gpu))

    all_y = []
    all_pred_base = []
    all_pred_new = []

    close_cnt = 0
    ocr_used = 0

    for fold in range(5):
        ids = np.load(f"oof/convnext_small_fold{fold}_ids.npy", allow_pickle=True)
        y   = np.load(f"oof/convnext_small_fold{fold}_y.npy")

        Pc  = np.load(f"oof/convnext_small_fold{fold}_probs.npy")
        Ps  = np.load(f"oof/swin_small_fold{fold}_probs.npy")

        P = args.conv_w * Pc + args.swin_w * Ps
        pred_base = P.argmax(1).copy()
        pred_new  = pred_base.copy()

        for i, img_id in enumerate(ids):
            # 3/7만 대상(너의 규칙이 적용되는 영역만)
            if pred_base[i] not in (3, 7):
                continue

            p3 = float(P[i, 3]); p7 = float(P[i, 7])

            # “조건 박빙” 필터: 여기 걸린 애들만 OCR로 흔들기
            if (abs(p3 - p7) >= args.close_margin) or ((p3 + p7) <= args.close_sum):
                continue

            close_cnt += 1

            key = str(img_id)
            if key in cache:
                title_text = cache[key].get("title", "")
                body_text  = cache[key].get("body", "")
            else:
                path = find_image(args.train_img_dir, img_id)
                if path is None:
                    continue
                img = cv2.imread(path)
                if img is None:
                    continue

                title = crop_region(img, "title")
                body  = crop_region(img, "body")

                title_text = " ".join(reader.readtext(title, detail=0, paragraph=True))
                body_text  = " ".join(reader.readtext(body,  detail=0, paragraph=True))

                cache[key] = {"title": title_text, "body": body_text}

            ocr_used += 1

            # 혼합 제목(입원+통원 같이 쓰는 경우) 처리:
            # 제목이 혼합이면 제목은 버리고 본문 키워드로만 결정
            title_has_3 = contains_any(title_text, KW3)
            title_has_7 = contains_any(title_text, KW7)
            body_has_3  = contains_any(body_text, KW3)
            body_has_7  = contains_any(body_text, KW7)

            decided = None
            if title_has_3 and title_has_7:
                if body_has_3 and not body_has_7:
                    decided = 3
                elif body_has_7 and not body_has_3:
                    decided = 7
            else:
                if (title_has_3 or body_has_3) and not (title_has_7 or body_has_7):
                    decided = 3
                elif (title_has_7 or body_has_7) and not (title_has_3 or body_has_3):
                    decided = 7

            if decided is not None:
                pred_new[i] = decided
            else:
                # bias7: “박빙 + 키워드 못찾음”이면 아주 약하게 7로 기울임
                if args.mode == "bias7":
                    p3 = float(P[i, 3]); p7 = float(P[i, 7])
                    if (abs(p3 - p7) < args.bias_margin) and ((p3 + p7) > args.bias_sum):
                        P2 = P[i].copy()
                        P2[7] += args.bias_eps
                        pred_new[i] = int(P2.argmax())

        all_y.append(y)
        all_pred_base.append(pred_base)
        all_pred_new.append(pred_new)

    y = np.concatenate(all_y)
    pred_base = np.concatenate(all_pred_base)
    pred_new  = np.concatenate(all_pred_new)

    f_base = f1_score(y, pred_base, average="macro")
    f_new  = f1_score(y, pred_new,  average="macro")

    changed = int((pred_base != pred_new).sum())

    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    with open(args.cache, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    print(f"[BASE] macro_f1={f_base:.6f}")
    print(f"[NEW ] macro_f1={f_new:.6f}  (delta={f_new-f_base:+.6f})")
    print(f"[INFO] close_cnt={close_cnt}, ocr_used={ocr_used}, changed={changed}")
    print(f"[INFO] mode={args.mode}, margin={args.close_margin}, sum={args.close_sum}, eps={args.bias_eps}, gpu={args.gpu}")

if __name__ == "__main__":
    main()
