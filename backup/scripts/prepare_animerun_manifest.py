"""
Tạo CSV (img1,img2,flow) cho AnimeRun_v2:
- Flow:   <root>/<split>/Flow/<topic>/{forward|backward}/ImageXXXX.flo
- Frames: <root>/<split>/Frame_Anime/<topic>/<variant>/<XXXX.png|ImageXXXX.png>

Ví dụ:
  python scripts/prepare_animerun_v2_manifest.py \
    --root /home/serverai/ltdoanh/optical_flow/data/AnimeRun_v2 \
    --split train --variant original \
    --out_csv data/animerun_v2_train.csv --direction forward
"""
import argparse, csv, os, glob, re

IMG_EXTS = [".png", ".jpg", ".jpeg"]
DIGITS = re.compile(r"(\d+)")  # match number in 'Image0012' or '0012'

def list_topics(flows_root: str):
    return [d for d in sorted(os.listdir(flows_root)) if os.path.isdir(os.path.join(flows_root, d))]

def infer_pad_and_prefix(frames_variant_dir: str):
    """Infer zero-padding and prefix ('Image' or empty) from any file in the folder."""
    cand = sorted(glob.glob(os.path.join(frames_variant_dir, "*")))
    for p in cand:
        base = os.path.splitext(os.path.basename(p))[0]
        m = DIGITS.search(base)
        if not m:
            continue
        return len(m.group(1)), base[:m.start()]  # (pad_width, prefix)
    return 4, "Image"

def frame_path(frames_variant_dir: str, idx: int, pad: int, prefix: str):
    stems = []
    if prefix: stems.append(f"{prefix}{idx:0{pad}d}")  # e.g., Image0001
    stems.append(f"{idx:0{pad}d}")                    # e.g., 0001
    stems.append(str(idx))                            # e.g., 1
    for stem in stems:
        for ext in IMG_EXTS:
            p = os.path.join(frames_variant_dir, stem + ext)
            if os.path.exists(p):
                return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--split', required=True, choices=['train','val','test'])
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--direction', default='forward', choices=['forward','backward','both'])
    ap.add_argument('--flows_dirname', default='Flow')
    ap.add_argument('--frames_dirname', default='Frame_Anime')
    ap.add_argument('--variant', default='original', help='original | clour 1 | clour 2 | clour 3 | clour 4')
    args = ap.parse_args()

    split_dir = os.path.join(args.root, args.split)
    flows_root = os.path.join(split_dir, args.flows_dirname)
    frames_root = os.path.join(split_dir, args.frames_dirname)

    assert os.path.isdir(flows_root), f"Flows root not found: {flows_root}"
    assert os.path.isdir(frames_root), f"Frames root not found: {frames_root}"

    topics = list_topics(flows_root)
    rows = []

    for topic in topics:
        topic_flow = os.path.join(flows_root, topic)
        topic_frames_variant = os.path.join(frames_root, topic, args.variant)
        if not os.path.isdir(topic_frames_variant):
            print(f"⚠️  Skip topic (frames variant not found): {topic} -> {topic_frames_variant}")
            continue

        pad, prefix = infer_pad_and_prefix(topic_frames_variant)

        def collect(dir_name: str, forward: bool):
            d = os.path.join(topic_flow, dir_name)
            if not os.path.isdir(d):
                return
            flos = sorted(glob.glob(os.path.join(d, 'Image*.flo')))
            for f in flos:
                stem = os.path.splitext(os.path.basename(f))[0]  # ImageXXXX
                m = DIGITS.search(stem)
                if not m:
                    continue
                idx = int(m.group(1))
                cur_idx = idx
                nxt_idx = idx + 1 if forward else idx - 1
                img1 = frame_path(topic_frames_variant, cur_idx, pad, prefix)
                img2 = frame_path(topic_frames_variant, nxt_idx, pad, prefix)
                if img1 and img2:
                    rows.append({
                        'img1': os.path.relpath(img1, args.root),
                        'img2': os.path.relpath(img2, args.root),
                        'flow': os.path.relpath(f, args.root),
                    })

        if args.direction in ('forward','both'):  collect('forward', True)
        if args.direction in ('backward','both'): collect('backward', False)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as wf:
        w = csv.DictWriter(wf, fieldnames=['img1','img2','flow'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"✅ Wrote {len(rows)} pairs → {args.out_csv}")

if __name__ == '__main__':
    main()



"""
# Train CSV (chọn biến thể ảnh 'original'; có thể đổi thành 'clour 1' ...):
python3 scripts/prepare_animerun_manifest.py \
  --root /home/serverai/ltdoanh/optical_flow/data/AnimeRun_v2 \
  --split train --variant "original" \
  --out_csv data/animerun_v2_train.csv --direction forward

# Val/Test CSV (nếu dùng test làm val):
python3 scripts/prepare_animerun_manifest.py \
  --root /home/serverai/ltdoanh/optical_flow/data/AnimeRun_v2 \
  --split test --variant "original" \
  --out_csv data/animerun_v2_val.csv --direction forward

"""