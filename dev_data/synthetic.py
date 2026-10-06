"""Render synthetic shelving units and a handheld pan across them, with exact ground truth.

Why: it is the only footage where the true count, every spine's size and every
title are known exactly, so the pipeline's counting, merging and measurement
can be scored without a tape measure. It does NOT replace the real capture:
real spines have fonts, wear, glare and depth this renderer does not.

Outputs (dev_data/synthetic/):
    unit_A.png, unit_B.png         the flat shelf faces (10 px per cm)
    frames/NNNN.jpg                the pan, 1280x720
    sweep.mjpeg                    the pan as a video Chrome can use as a fake camera
    ground_truth.json              every book: unit, shelf, title, author, height, thickness, flat?

Usage:  python dev_data/synthetic.py
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np

from library_claim.stages.scale import render_marker

PX = 25  # pixels per cm on the flat faces (sharp enough that frames are not upscaled mush)
MARKER_CM = 10.0  # fits wholly inside the first row of close-up frames
OUT = Path(__file__).parent / "synthetic"

# Real, widely sold books so identification and pricing can run against live sources.
BOOKS = [
    ("Sapiens", "Yuval Noah Harari", "Vintage"), ("Atomic Habits", "James Clear", "Random House"),
    ("The Alchemist", "Paulo Coelho", "HarperCollins"), ("Nineteen Eighty-Four", "George Orwell", "Penguin"),
    ("Animal Farm", "George Orwell", "Penguin"), ("The God of Small Things", "Arundhati Roy", "Penguin"),
    ("The White Tiger", "Aravind Adiga", "HarperCollins"), ("Midnight's Children", "Salman Rushdie", "Vintage"),
    ("A Suitable Boy", "Vikram Seth", "Penguin"), ("The Psychology of Money", "Morgan Housel", "Jaico"),
    ("Thinking, Fast and Slow", "Daniel Kahneman", "Penguin"), ("The Hobbit", "J. R. R. Tolkien", "HarperCollins"),
    ("Pride and Prejudice", "Jane Austen", "Penguin"), ("To Kill a Mockingbird", "Harper Lee", "Arrow"),
    ("The Great Gatsby", "F. Scott Fitzgerald", "Penguin"), ("Brave New World", "Aldous Huxley", "Vintage"),
    ("The Catcher in the Rye", "J. D. Salinger", "Penguin"), ("Wings of Fire", "A. P. J. Abdul Kalam", "Universities Press"),
    ("Ikigai", "Hector Garcia", "Penguin"), ("Rich Dad Poor Dad", "Robert Kiyosaki", "Plata"),
    ("The Kite Runner", "Khaled Hosseini", "Bloomsbury"), ("Educated", "Tara Westover", "Windmill"),
    ("Becoming", "Michelle Obama", "Penguin"), ("Deep Work", "Cal Newport", "Piatkus"),
    ("Clean Code", "Robert C. Martin", "Pearson"), ("The Pragmatic Programmer", "David Thomas", "Addison-Wesley"),
    ("Dune", "Frank Herbert", "Hodder"), ("Norwegian Wood", "Haruki Murakami", "Vintage"),
    ("The Namesake", "Jhumpa Lahiri", "HarperCollins"), ("Train to Pakistan", "Khushwant Singh", "Penguin"),
    ("The Palace of Illusions", "Chitra Banerjee Divakaruni", "Picador"), ("Sophie's World", "Jostein Gaarder", "Phoenix"),
    ("Man's Search for Meaning", "Viktor E. Frankl", "Rider"), ("The Silent Patient", "Alex Michaelides", "Orion"),
    ("Where the Crawdads Sing", "Delia Owens", "Corsair"), ("The Midnight Library", "Matt Haig", "Canongate"),
    ("Shantaram", "Gregory David Roberts", "Abacus"), ("The Inheritance of Loss", "Kiran Desai", "Penguin"),
    ("Long Walk to Freedom", "Nelson Mandela", "Abacus"), ("The Body", "Bill Bryson", "Black Swan"),
    ("A Brief History of Time", "Stephen Hawking", "Bantam"), ("The Selfish Gene", "Richard Dawkins", "Oxford"),
    ("Guns, Germs, and Steel", "Jared Diamond", "Vintage"), ("The Lean Startup", "Eric Ries", "Portfolio"),
    ("Zero to One", "Peter Thiel", "Virgin"), ("The Old Man and the Sea", "Ernest Hemingway", "Vintage"),
    ("Of Mice and Men", "John Steinbeck", "Penguin"), ("Fahrenheit 451", "Ray Bradbury", "HarperCollins"),
    ("Life of Pi", "Yann Martel", "Canongate"), ("The Book Thief", "Markus Zusak", "Black Swan"),
    ("Malgudi Days", "R. K. Narayan", "Penguin"), ("Gitanjali", "Rabindranath Tagore", "Fingerprint"),
    ("The Immortals of Meluha", "Amish Tripathi", "Westland"), ("Five Point Someone", "Chetan Bhagat", "Rupa"),
    ("Homo Deus", "Yuval Noah Harari", "Vintage"), ("Range", "David Epstein", "Macmillan"),
    ("Outliers", "Malcolm Gladwell", "Penguin"), ("Quiet", "Susan Cain", "Penguin"),
    ("The Power of Habit", "Charles Duhigg", "Random House"), ("Ego Is the Enemy", "Ryan Holiday", "Profile"),
]


def _spine(height_cm, thick_cm, title, author, publisher, color, legible):
    """One spine image (upright), text running bottom-to-top like most English spines."""
    h, w = int(height_cm * PX), int(thick_cm * PX)
    spine = np.full((h, w, 3), color, np.uint8)
    cv2.rectangle(spine, (0, 0), (w - 1, h - 1), tuple(int(c * 0.6) for c in color), 1)
    if not legible:
        return spine
    ink = (20, 20, 20) if sum(color) > 380 else (245, 245, 245)
    # Draw text horizontally on a canvas the length of the spine, then rotate it onto the spine.
    canvas = np.full((w, h, 3), color, np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX

    def put(text, start, end, max_scale):
        """Fit text inside [start, end] of the spine's length and inside its thickness."""
        room = int((end - start) * h) - 4
        (tw, th), _ = cv2.getTextSize(text, font, 1.0, 1)
        scale = min(max_scale, room / tw, (w * 0.7) / th)
        if scale < 0.25:
            return
        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
        cv2.putText(canvas, text, (int(start * h) + 2, (w + th) // 2), font, scale, ink, 1, cv2.LINE_AA)

    put(title.upper(), 0.04, 0.60, 1.0)
    put(author, 0.63, 0.86, 0.8)
    put(publisher, 0.88, 0.99, 0.6)
    return np.ascontiguousarray(np.rot90(canvas, 1))


def build_unit(name, books, rng, width_cm=50, shelf_heights=(34, 34, 34, 30)):
    """A shelving unit face with books packed on each shelf; returns image and ground-truth rows."""
    height_cm = sum(shelf_heights) + 3 * (len(shelf_heights) + 1)
    face = np.full((height_cm * PX, width_cm * PX, 3), (60, 95, 140), np.uint8)  # wood
    truth = []
    per_shelf = -(-len(books) // len(shelf_heights))
    chunks = [list(books[i:i + per_shelf]) for i in range(0, len(books), per_shelf)]
    leftovers = 0
    y = 3
    for shelf_index, shelf_h in enumerate(shelf_heights, start=1):
        pool = chunks[shelf_index - 1] if shelf_index - 1 < len(chunks) else []
        inner_bottom = y + shelf_h
        face[y * PX:inner_bottom * PX, 2 * PX:(width_cm - 2) * PX] = (40, 55, 70)  # shelf back
        x = 2 + (MARKER_CM + 2 if shelf_index == 1 else 0) + 0.5
        position = 0
        stack_done = False
        while pool:
            if not stack_done and shelf_index in (2, 4) and len(pool) <= 3:
                # A stack of books lying flat: their spines face out horizontally.
                stack_y = inner_bottom
                for _ in range(3):
                    if not pool:
                        break
                    title, author, publisher = pool.pop(0)
                    length, thick = rng.uniform(18, 24), rng.uniform(1.8, 3.2)
                    if x + length > width_cm - 2 or stack_y - thick < y + 2:
                        pool.insert(0, (title, author, publisher))
                        break
                    color = tuple(int(c) for c in rng.integers(30, 230, 3))
                    upright = _spine(length, thick, title, author, publisher, color, True)
                    flat = np.ascontiguousarray(np.rot90(upright, -1))
                    top = int((stack_y - thick) * PX)
                    face[top:top + flat.shape[0], int(x * PX):int(x * PX) + flat.shape[1]] = flat
                    position += 1
                    truth.append({"unit": name, "shelf": shelf_index, "position": position, "title": title, "author": author,
                                  "publisher": publisher, "height_cm": round(length, 1), "thickness_cm": round(thick, 1),
                                  "orientation": "flat", "legible": True})
                    stack_y -= thick
                x += 25.5
                stack_done = True
                continue
            title, author, publisher = pool[0]
            height, thick = rng.uniform(17.5, min(31.0, shelf_h - 1.5)), rng.uniform(1.4, 4.2)
            if x + thick > width_cm - 2:
                break
            pool.pop(0)
            legible = rng.random() > 0.12  # ~12% blank spines: must be counted but not titled
            color = tuple(int(c) for c in rng.integers(30, 230, 3))
            spine = _spine(height, thick, title, author, publisher, color, legible)
            top = int((inner_bottom - height) * PX)
            face[top:top + spine.shape[0], int(x * PX):int(x * PX) + spine.shape[1]] = spine
            position += 1
            truth.append({"unit": name, "shelf": shelf_index, "position": position,
                          "title": title if legible else "", "author": author if legible else "",
                          "publisher": publisher if legible else "", "height_cm": round(height, 1),
                          "thickness_cm": round(thick, 1), "orientation": "upright", "legible": legible})
            x += thick + rng.uniform(0.0, 0.4)
        leftovers += len(pool)
        y = inner_bottom + 3
    # Marker on the top shelf, left side.
    marker = render_marker(0, 400)
    side = int(MARKER_CM * PX)
    scaled = cv2.resize(marker, None, fx=side / 400, fy=side / 400, interpolation=cv2.INTER_NEAREST)
    pad = (scaled.shape[0] - side) // 2
    my, mx = 5 * PX, 3 * PX  # marker's top-left at (3, 5) cm: inside the first row of frames
    face[my - pad:my - pad + scaled.shape[0], mx - pad:mx - pad + scaled.shape[1]] = scaled[..., None]
    return face, truth, leftovers


def pan_frames(faces, out_dir, rng, size=(1280, 720)):
    """A handheld pan down each unit, framed like a phone held ~70 cm from a narrow bookcase.

    Each view is 1.8x the unit's width (90 x 50 cm for a 50 cm unit), so whole
    shelf rows (34 cm) fit inside frames, as in a real sweep. Consecutive views
    overlap ~70% vertically (a 0.7 s keyframe interval at a slow downward pan),
    which puts every row wholly inside at least one frame.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = size
    index = 0
    for face in faces:
        fh, fw = face.shape[:2]
        view_w = int(fw * 1.8)
        view_h = int(view_w * h / w)
        left = (fw - view_w) / 2
        for top in np.arange(-view_h * 0.1, fh - view_h * 0.9, view_h * 0.3):
            jitter = rng.uniform(-0.03, 0.03, 8).reshape(4, 2) * [view_w, view_h]
            src = np.float32([[left, top], [left + view_w, top], [left + view_w, top + view_h], [left, top + view_h]]) + jitter.astype(np.float32)
            tilt = rng.uniform(-0.05, 0.05) * w
            dst = np.float32([[0 + tilt, 0], [w, 0 + abs(tilt) * 0.3], [w - tilt, h], [0, h]])
            frame = cv2.warpPerspective(face, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(200, 200, 200))
            if rng.random() < 0.1:
                frame = cv2.blur(frame, (25, 3))  # motion blur, as when panning too fast
            cv2.imwrite(str(out_dir / f"{index:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            index += 1
        # Walking to the next unit: a few frames of blank wall break the tracking chain.
        for _ in range(3):
            cv2.imwrite(str(out_dir / f"{index:04d}.jpg"), np.full((h, w, 3), (205, 210, 215), np.uint8))
            index += 1
    return index


def main(seed=11):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    OUT.mkdir(exist_ok=True)
    books = BOOKS.copy()
    rng.shuffle(books)
    face_a, truth_a, left = build_unit("A", books[:34], rng)
    face_b, truth_b, left_b = build_unit("B", books[34:], rng)
    cv2.imwrite(str(OUT / "unit_A.png"), face_a)
    cv2.imwrite(str(OUT / "unit_B.png"), face_b)
    frames = pan_frames([face_a, face_b], OUT / "frames", rng)
    truth = truth_a + truth_b
    (OUT / "ground_truth.json").write_text(json.dumps({
        "book_count": len(truth), "legible": sum(b["legible"] for b in truth), "marker_cm": MARKER_CM,
        "unit_faces_cm": [list(np.array(face_a.shape[1::-1]) / PX), list(np.array(face_b.shape[1::-1]) / PX)],
        "books": truth, "frames": frames, "unshelved": left + left_b,
    }, indent=1), encoding="utf-8")
    # Chrome's fake camera loops an MJPEG file (compressed, unlike .y4m which was 2 GB here).
    # Each view is held for 1.2 s at 10 fps, like a person pausing on each part of the shelf.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "0.83", "-i", str(OUT / "frames" / "%04d.jpg"),
                    "-vf", "fps=10", "-q:v", "3", str(OUT / "sweep.mjpeg")], check=True)
    print(f"{len(truth)} books ({sum(b['legible'] for b in truth)} legible), {frames} frames -> {OUT}")


if __name__ == "__main__":
    main()
