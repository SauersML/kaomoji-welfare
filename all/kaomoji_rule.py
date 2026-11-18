"""Which entries of the full face set are literally kaomoji.

A kaomoji is a Japanese-style face read upright: eyes and mouth in one horizontal row, usually inside brackets,
sometimes with arms or decorations around it. The rule drops, in this order (first reason that fires):

  western      a Western sideways emoticon (the face reads rotated 90 degrees): somewhere in the entry an eye
               ':' ';' '=' sits next to an optional nose (- ' ^ o ~ ,) and a mouth ( ( ) [ ] { } | / \\ D P p O o 0 3 * @ $ & < > c C I J L S ),
               in either order, e.g. ":-(", "(-:", ">:(", "[:|]", ":'(", "(:3 _ )=". Kept anyway when the face
               inside the brackets reads upright: it contains a non-ASCII character, or (after trimming a sweat mark
               ';' or ':' at its ends) its first and last characters are the same or mirror images, as in "(;_;)",
               "(^_^;)", "(=^_^=)".
  single       the bracket holds fewer than two visible characters, that character is ASCII, and the entry has no
               pair of eyes elsewhere: "(a)", "(Q)", "c[_]", "( , )", "( /)u(\\ )". (A single non-ASCII glyph that is itself
               a face, such as the ツ of "¯\\_(ツ)_/¯", is kept, and so is a face holding a cup, "=＾●⋏●＾=⊃c[_]❀".)
  punctuation  apart from brackets the entry is one repeated mark from . , _ - ' " ` ~ : no eyes, e.g. "(..)", "(__)", "( _ _ )".
  word         letters spell a word, either
               - outside the brackets: after NFKC, two or more Latin letters in a row ("Lᵒᵛᵉᵧₒᵤ", "ＧＯＯＤ", "ⒽⓤⒼ"),
                 or three or more kana, kanji, hangul, Greek or Cyrillic letters in a row ("いただきます", "ﾖｼﾖｼ",
                 "ﾅﾃﾞﾅﾃﾞ", "βακα"), counting only runs with at least two different letters, case ignored (ΞΞΞ motion
                 lines, 炎炎炎 fire, 人人人 frames, Ｚｚｚ sleep are drawings); letters used as arms or hands (ﾉ ノ つ っ
                 ヽ ヾ) do not count, nor ™ or the butterfly εїз; or
               - as the whole content of a bracket: two or more letters and nothing else, with no two equal or mirrored
                 letters that could be eyes: "(語彙力)", "(尊い)", "[ＤЁАЯ]", "(ｷﾘｯ)" go, "(ΦωΦ)", "(のしの)", "(ᵘʷᵘ)" stay.
               (Entries with two or more ASCII letters in a row anywhere were already removed when the set was built.)

Brackets pair loosely (any closing bracket closes any opening one, since faces mix them, as in "(╯°□°）╯"). With no
balanced pair, the face runs from the first opening bracket to the end ("(ಥ_ಥ"); only what comes before it is outside.
For the western and single tests, the face inside the brackets is the content of the first balanced bracket pair; for an
entry with no balanced pair (e.g. "(ಥ_ಥ", ":-("), it is the whole entry with leading and trailing brackets removed.
Run as a script: unit examples, counts, and the kept/dropped lists (results/main/kaomoji_keep.tsv).
"""
import re, sys, unicodedata

OPEN = "(（[［{｛ʕ༼【〔⁽₍〈«꒰⸨"
CLOSE = ")）]］}｝ʔ༽】〕⁾₎〉»꒱⸩"
PAIR = dict(zip(OPEN, CLOSE))
EYE, NOSE = r"[:;=]", r"[-'^o~,_]?"
MOUTH_R = r"[()\[\]{}|/\\DPpOo03*@$&<>cCIJLS]"   # after the eyes, e.g. :-(  :P  =3
MOUTH_L = r"[()\[\]{}|/\\DdOo<>cC]"             # before the eyes, e.g. (-:  D:
WEST = re.compile(EYE + NOSE + MOUTH_R + "|" + MOUTH_L + NOSE + EYE + "|" + r"[8Xx*%B][-^']" + MOUTH_R + "|" + r"[>}][xX8]" + MOUTH_R)
MIRROR = {"(": ")", "[": "]", "{": "}", "<": ">", ">": "<", "/": "\\", "\\": "/", "d": "b", "b": "d", "p": "q", "q": "p", "'": "`", "`": "'"}
BARE = set(".,_-'\"`~")


def face_inner(s):
    """Content of the first balanced bracket pair, else the entry with outer brackets stripped."""
    for i, c in enumerate(s):
        if c in OPEN:
            st = [c]
            for j in range(i + 1, len(s)):
                if s[j] in OPEN: st.append(s[j])
                elif s[j] in CLOSE:
                    if PAIR[st[-1]] != s[j]: break
                    st.pop()
                    if not st:
                        if s[i + 1:j].strip(): return s[i + 1:j]
                        break
    return s.strip(OPEN + CLOSE)


def groups(s):
    """(start, end) of every balanced bracket pair; any closing bracket closes any opening one, since faces mix them,
    as in "(╯°□°）╯"; unmatched brackets belong to none."""
    st, out = [], []
    for i, c in enumerate(s):
        if c in OPEN: st.append(i)
        elif c in CLOSE and st: out.append((st.pop(), i))
    return out


def outside(s):
    """The entry with every balanced bracket pair and its content blanked (unmatched brackets blanked alone).
    With no balanced pair, the face runs from the first opening bracket to the end, as in "(ಥ_ಥ", "(ΘεΘ"; only what comes
    before that bracket is outside (nothing, if the entry has no opening bracket)."""
    g = groups(s)
    if not g:
        i = next((i for i, c in enumerate(s) if c in OPEN), 0); return s[:i]
    o = [" " if c in OPEN + CLOSE else c for c in s]
    for a, b in g: o[a:b + 1] = " " * (b + 1 - a)
    return "".join(o)


NO_EYE = set(OPEN + CLOSE + ":_") | {" ", "\u3000"}
NO_TOUCH = set("=-|.,~'")   # may be eyes, but not with nothing between them (==, --, ||)


def eye_pair(s, skip=()):
    """Two equal or mirrored eye characters in one row with 0-9 characters between them and no bracket between."""
    for i, x in enumerate(s):
        if x in NO_EYE or x in skip: continue
        for k in range(1, 11):
            j = i + k
            if j >= len(s): break
            if s[j] in OPEN + CLOSE: break
            if (s[j] == x or MIRROR.get(x) == s[j]) and not (k == 1 and x in NO_TOUCH): return True
    return False


LOOKALIKE = str.maketrans({c: "-" for c in "‐‑‒–—―−﹣－˗"} | {"¦": "|"})   # dashes and the broken bar read as ASCII: ":‑(" is ":-("


UPRIGHT_TEAR = re.compile(r"[;:/\\]_[;:]|_ +_")   # crying with an underscore mouth, (/_;) (;_:) (:_;); bowing, (o;_ _)o


def upright(f, inner):
    return any(ord(c) > 127 for c in inner.translate(LOOKALIKE)) or eye_pair(f) or bool(UPRIGHT_TEAR.search(inner)) or bool(UPRIGHT_TEAR.search("".join(inner.split())))


JOIN = set("ﾞ゛ーｰっッｯぁぃぅぇぉゃゅょァィゥェォャュョｧｨｩｪｫｬｭｮ")


def has_eyes(g):
    """Two equal or mirrored letters: a pair of eyes (ΦωΦ, のしの, uωu人, 灬ºωº灬, 一一, ΘΘ)."""
    return any(g[j] == x or MIRROR.get(x) == g[j] for i, x in enumerate(g) for j in range(i + 1, len(g)))


ARMS = set("つっノﾉヽヾゞ")   # letters used as arms or hands inside a bracket, as in "(つ益)つ"


def word_letter(c):
    o = ord(c)
    return (0x3041 <= o <= 0x3096 or 0x30A1 <= o <= 0x30FA or 0xFF66 <= o <= 0xFF9D or 0x4E00 <= o <= 0x9FFF
            or 0xAC00 <= o <= 0xD7A3 or 0x0370 <= o <= 0x03FF or 0x0400 <= o <= 0x04FF)


def letter_run(s, latin):
    """Longest run of word letters with at least two different letters (ΞΞΞ, 炎炎炎, 人人人 are drawings, not words)."""
    best, run = 0, ""
    for c in (unicodedata.normalize("NFKC", s) if latin else s) + " ":
        ok = (c.isascii() and c.isalpha()) if latin else ((word_letter(c) and c not in ARMS) or (run and c in JOIN))
        if ok: run += c
        else:
            if len({x.lower() for x in run}) >= 2: best = max(best, len(run))   # per letter: "ΣΣΣΣ".lower() ends in a final sigma   # Ｚｚｚ (sleep) and oＯ (bubbles) are drawings
            run = ""
    return best


# words spelled in look-alike letters from several scripts, which the letter-run test cannot see (found in the hand checks):
# "bye" (βyё, вуe), "hello" (нєℓℓσ), "chu" (cнϋ), "haha" (ʱªʱª), "good night" (Gɵɵd ɳɩɠɧt), "thanks" (ᵗʱᵃᵑᵏ), "what" (ωҺa),
# "internet", "heh", "ugh", "why" (ʷʱʸ), "miss you" (Ⓨσu), "forever", "love" (ιоνё), "suki" (СНЦ), and words written upside
# down after a flipped table (ǝ ɹ ʞ ɯ ᴉ ɐ ʇ are turned letters)
SPELLED = ("βyё", "βｙё", "вуe", "нєℓℓσ", "cнϋ", "ʱªʱª", "ɵɵd", "ɳɩɠɧ", "ᵗʱᵃᵑᵏ", "ωҺa", "ɪɴᴛᴇʀɴᴇᴛ", "ʜᴇʜ", "ᴜɢʜ", "ʷʱʸ",
           "Ⓨσu", "Ⓨσư", "ʳᵧ", "СНЦ", "снц", "ιоνё", "ǝ", "ɹ", "ʞ", "ɯ", "ᴉ", "ɐ")


def why_not(f):
    """None if f is a kaomoji, else the reason it is dropped."""
    inner = face_inner(f); vis = "".join(inner.split())
    if WEST.search("".join(f.split()).translate(LOOKALIKE)) and not upright(f, inner): return "western"
    if len(vis) < 2 and (not vis or vis.isascii()) and not eye_pair(f, BARE): return "single"
    rest = "".join(c for c in f if not c.isspace() and c not in OPEN + CLOSE)
    if rest and len(set(rest)) == 1 and rest[0] in BARE: return "punctuation"
    out = outside(f).replace("εїз", " ").replace("™", " ").replace("℡", " ").replace("三二一", " ")   # εїз is a butterfly; ™ ℡ are signs, not letters; 三二一 is speed lines
    if letter_run(out, True) >= 2 or letter_run(out, False) >= 3 or any(w in out for w in SPELLED): return "word"
    for a, b in groups(f):   # a bracket holding only letters and no pair of eyes: "(語彙力)", "[ＤЁАЯ]", "(ｷﾘｯ)"
        g = "".join(c for c in unicodedata.normalize("NFKC", f[a + 1:b]) if not c.isspace() and c != "…" and c not in ARMS)
        if len(g) >= 2 and all((c.isascii() and c.isalpha()) or word_letter(c) or c in JOIN for c in g) and not has_eyes(g): return "word"
    return None


KEEP = ["(^_^)", "(╥﹏╥)", "(;_;)", "(^_^;)", "(−_−;)", "(ーー;)", "(=^･^=)", "(-_-)", "(T_T)", "(>_<)", "(u_u)", "(^^)", "(・・?)",
        "¯\\_(ツ)_/¯", "(ಥ_ಥ", "(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧", "ʕ•ᴥ•ʔ", "( ͡° ͜ʖ ͡°)", "(;O;)", "d(^_^)b", "＼(＾▽＾)／", "(╯°□°）╯︵ ┻━┻", "(o.o)", "(!_!)",
        "ヽ(ﾟ∀ﾟ)ﾉ", "( •̀ ω •́ )ﾑﾝ", "⊂二二二(　＾ω＾)二⊃", "(*’▽’)ノ＾—==ΞΞΞ☆", "炎炎炎炎☆┣o(･ω･ )", "(;^_^A", "(-_-;)", "(*^_^*)", "m(_ _)m", "φ(..)",
        "(ΦωΦ)", "(ΘεΘ", "(のしの)", "(ᵘʷᵘ)", "(ＴoＴ)", "(бвб)", "(ﾉ･o･)ﾉ", "@(o･ｪ･)@", "(σ･ω･)σ", "(灬ºωº灬)", "(人ФｴФ)", "(uωu人)", "(一一)", "m(ΘΘ)m", "(つ益)つ︵┻━┻", "¯\\(º o)/¯", "⊂(( っ☉ω☉)っ　　～εїз", "=＾●⋏●＾=⊃c[_]❀", "™(✿◠‿◠)™", "(V)o￥o(V)", "!(ﾉ｀◎´)ﾉミρ┳┷┳ﾟσ(￣○￣;;)", "!!!!!(o≧ω≦)○))ｏ(＿＿*)Ｚｚｚ", "(/_;)", "(;_:)", "(:_;)", "( / _ ; )", "(o;_ _)o", "℡ψ (｀ ∇´) ψ℡", "(’ω’ ⊃ )⊃三二一"]
DROP = {"(a)": "single", ":‑(": "western", ":(¦)": "western", "(-:": "western", ":-(": "western", ": (": "western", ">:(": "western", ">:-(": "western", "[:|]": "western",
        ":'(": "western", "(:I": "western", "(:3 _ )=": "western", ":=(": "western", "(..)": "punctuation", "(__)": "punctuation",
        "( _ _ )": "punctuation", "c[_]": "single", "( , )": "single", "(Q)": "single", "Lᵒᵛᵉᵧₒᵤ(•ᴗ•♡)": "word",
        "いただきます(＾∀＾)": "word", "(*´з｀)ＧＯＯＤ ＤЯЁΑΜ(´ε｀*)": "word", "( ̄ー ̄)ﾉ ﾖｼﾖｼ": "word", "٩(๑•◡-๑)۶ⒽⓤⒼ": "word",
        "(語彙力)": "word", "┬┴┬┴┤ᵒᵏ (･_├┬┴┬┴": "word", "(ﾟ∀ﾟ)イラッシャイ!": "word", "ʰᵘʰ (ꐦ○_○）✧": "word", "(尊い)": "word", "(´ω｀)ﾉﾅﾃﾞﾅﾃﾞ": "word", "*|:^)(.)(...)": "single", "[ＤЁАЯ]q･∀･o)": "word", "(ｷﾘｯ)(・ωｰ)": "word", "(:˒[￣] ᵍᵒᵒᵈ ⁿᶦᵍʰᵗ": "word", "вуe вуe☆ (・ω・`p[": "word", "ヾ(ﾟεÅ) βyё βyё☆彡": "word", "(╯°□°)╯︵ ʞɹoʍ": "word"}
KEEP += ["ΣΣΣΣ（０＜）"]


def check_units():
    bad = [(f, why_not(f)) for f in KEEP if why_not(f) is not None] + [(f, why_not(f), r) for f, r in DROP.items() if why_not(f) != r]
    assert not bad, bad


if __name__ == "__main__":
    import csv, collections, random
    sys.path.insert(0, ".")
    from scen import KSET
    check_units()
    assert all(why_not(k) is None for k in KSET), [k for k in KSET if why_not(k)]
    tsv = sys.argv[1] if len(sys.argv) > 1 else "results/main/faces_all.tsv"
    import pandas as pd
    faces = pd.read_csv(tsv, sep="\t", keep_default_na=False)["face"].tolist()
    rs = [why_not(f) for f in faces]
    print("entries", len(faces), "kept", rs.count(None), dict(collections.Counter(r for r in rs if r)))
    pd.DataFrame(dict(face_idx=range(len(faces)), face=faces, kaomoji=[int(r is None) for r in rs], reason=[r or "" for r in rs])).to_csv(
        "results/main/kaomoji_keep.tsv", sep="\t", index=False)
    random.seed(0)
    kept = [f for f, r in zip(faces, rs) if r is None]; drop = [(f, r) for f, r in zip(faces, rs) if r]
    print("KEPT SAMPLE"); [print(" ", f) for f in random.sample(kept, 200)]
    print("DROPPED SAMPLE"); [print(" ", r, f) for f, r in random.sample(drop, 100)]
