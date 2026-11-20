"""Generate synthetic handwriting-style test pages with known transcriptions.

These exercise the Phase 3 pipeline (segmentation, blank answers, illegible regions, multi-page
answers, out-of-order answers, preserved misspellings) while the real gold set (task 1.6) is
being collected. Font-rendered text is much easier to read than real handwriting, so accuracy
numbers from these pages are NOT a substitute for the gold set.

Usage:  python scripts/make_synthetic_pages.py   (writes tests/fixtures/synthetic/)
"""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic"
FONTS = {
    "noteworthy": ("/System/Library/Fonts/Noteworthy.ttc", 0),
    "bradley": ("/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf", 0),
    "marker": ("/System/Library/Fonts/MarkerFelt.ttc", 0),
}
PAGE = (1240, 1754)  # A4 at 150 dpi
MARGIN, LINE_H = 110, 62

QUESTIONS = [
    {"question_id": "Q1", "prompt": "Define photosynthesis."},
    {"question_id": "Q2", "prompt": "Name the green pigment in leaves and say where it is found."},
    {"question_id": "Q3", "prompt": "Why do plants need water?"},
]

# Each block: (label, text, blur) — blur=True renders the text, then smudges it unreadably.
ITEMS = [
    {
        "id": "syn_01_clean", "font": "noteworthy", "rotate": 0.0,
        "pages": [[
            ("1.", "Photosynthesis is the process by which green plants make their own food using "
                   "sunlight, carbon dioxide and water. Oxygen is released as a by-product.", False),
            ("2.", "The green pigment is chlorophyll. It is found in the chloroplasts of leaf cells.", False),
            ("3.", "Plants need water for photosynthesis and to carry minerals from the roots to the leaves.", False),
        ]],
        "expect": {"blank": [], "illegible": []},
    },
    {
        "id": "syn_02_messy_illegible", "font": "bradley", "rotate": 1.8,
        "pages": [[
            ("Ans 1", "Photosynthesys is how plants make food from sunlite and carbon dioxide.", False),
            ("Ans 2", "The pigment is chlorophyl and it is present inside the chloroplast.", True),
            ("Ans 3", "Water keeps the plant cells firm and is used in photosynthesys.", False),
        ]],
        "expect": {"blank": [], "illegible": ["Q2"]},
    },
    {
        "id": "syn_03_blank_out_of_order", "font": "marker", "rotate": -1.0,
        "pages": [[
            ("Q2)", "Chlorophyll, which is found in chloroplasts.", False),
            ("Q1)", "Plants use light energy to turn carbon dioxide and water into glucose.", False),
            ("Q3)", "", False),
        ]],
        "expect": {"blank": ["Q3"], "illegible": []},
    },
    {
        "id": "syn_04_two_pages", "font": "noteworthy", "rotate": 0.6,
        "pages": [
            [("1.", "Photosynthesis happens in the leaves. Light energy is absorbed by chlorophyll "
                    "and used to split water molecules. Carbon dioxide from the air is then", False)],
            [("", "combined with hydrogen to make glucose, and oxygen is given out.", False),
             ("2.", "Chlorophyll, in the chloroplasts.", False),
             ("3.", "To make food and to transport nutrients.", False)],
        ],
        "expect": {"blank": [], "illegible": []},
    },
]


def wrap(draw, text, font, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + ([line] if line else [])


def render_page(blocks, font, rotate, rng):
    img = Image.new("RGB", PAGE, (250, 248, 240))
    d = ImageDraw.Draw(img)
    for y in range(MARGIN + LINE_H - 8, PAGE[1] - MARGIN, LINE_H):  # ruled lines
        d.line([(60, y), (PAGE[0] - 60, y)], fill=(205, 215, 235), width=2)
    y = MARGIN
    for label, text, blur in blocks:
        x0 = MARGIN
        if label:
            d.text((x0, y), label, font=font, fill=(25, 35, 90))
            x0 += int(d.textlength(label, font=font)) + 20
        lines = wrap(d, text, font, PAGE[0] - MARGIN - x0) if text else []
        top = y
        for i, line in enumerate(lines or [""]):
            jitter = rng.randint(-2, 2)
            d.text((x0 if i == 0 or not label else x0, y + jitter), line, font=font, fill=(25, 35, 90))
            y += LINE_H
        if blur and lines:
            box = (x0 - 10, top - 5, PAGE[0] - MARGIN + 10, y)
            region = img.crop(box).filter(ImageFilter.GaussianBlur(9))
            img.paste(region, box)
        y += LINE_H // 2
    img = img.rotate(rotate, resample=Image.BICUBIC, fillcolor=(250, 248, 240))
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    items = []
    for item in ITEMS:
        path, index = FONTS[item["font"]]
        font = ImageFont.truetype(path, 40, index=index)
        files, transcription = [], {q["question_id"]: [] for q in QUESTIONS}
        current = None
        for n, blocks in enumerate(item["pages"], start=1):
            name = f"{item['id']}_p{n}.jpg"
            render_page(blocks, font, item["rotate"], rng).save(OUT / name, quality=88)
            files.append(name)
            for label, text, _ in blocks:
                if label:
                    digit = next(ch for ch in label if ch.isdigit())
                    current = f"Q{digit}"
                transcription[current].append(text)
        items.append({
            "id": item["id"],
            "files": files,
            "transcription": {q: " ".join(t for t in parts if t) for q, parts in transcription.items()},
            "expect": item["expect"],
            "teacher_marks": None,
        })
    manifest = {
        "description": "SYNTHETIC font-rendered pages for pipeline testing. Not a substitute for the real "
                       "gold set (SENTINELEDU_TASKS.md 1.6). Regenerate with scripts/make_synthetic_pages.py.",
        "synthetic": True,
        "questions": QUESTIONS,
        "items": items,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {sum(len(i['files']) for i in items)} pages to {OUT}")


if __name__ == "__main__":
    main()
