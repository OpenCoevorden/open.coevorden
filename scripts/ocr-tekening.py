#!/usr/bin/env python3
# =============================================================================
# ocr-tekening.py — OCR voor bouwtekeningen (meerdere leesrichtingen + poort)
# =============================================================================
#
# WAT DOET DIT SCRIPT?
#   Leest één PDF, rastert de pagina('s) en haalt tesseract er in meerdere
#   LEESRICHTINGEN overheen. Bouwtekeningen hebben tekst die deels horizontaal
#   staat (annotaties in het veld) en deels 90° gedraaid (het titelblok).
#   Eén OCR-pass mist per definitie de helft.
#
#   Print de opgeschoonde tekst naar stdout; voortgang/diagnose naar stderr.
#   Aanroep:  python3 scripts/ocr-tekening.py tekening.pdf > tekst.txt
#
# -----------------------------------------------------------------------------
# WAAROM DE TWEE FILTERS?
# -----------------------------------------------------------------------------
#   1) CONFIDENCE-FILTER (OCR_MIN_CONF)
#      Tesseract levert per woord een betrouwbaarheidsscore. We vragen daarom
#      TSV-uitvoer i.p.v. platte tekst en gooien woorden onder de drempel weg.
#      Dat haalt het gros van de ruis eruit die op een tekening ontstaat: OCR
#      leest arceringen, maatlijnen en symbolen als losse letters.
#
#   2) WOORDENBOEK-POORT (OCR_MIN_DICT_RATIO)  ← de belangrijkste
#      Een pass in de VERKEERDE richting levert geen leegte op, maar
#      overtuigend ogende onzin: "Ausführungsplanung" ondersteboven wordt
#      "BunuejdsBunsyunysny", en dat haalt een hoge confidence. Zulke ruis
#      overleeft filter 1 dus moeiteloos.
#      Daarom meten we per richting welk aandeel van de regels minstens één
#      woord uit een Duits/Nederlands/Engels woordenboek bevat. Op de
#      testtekening: 0,85 (0°) en 0,69 (90°) tegenover 0,05 (180°) en
#      0,07 (270°). Alles onder de drempel wordt in zijn geheel verworpen.
#      Zo hoef je de richtingen niet per document te tunen: zet ze allemaal
#      aan en het script kiest zelf.
#
# -----------------------------------------------------------------------------
# AFHANKELIJKHEDEN (installeer via de workflow)
#   poppler-utils                → pdftoppm (rasteren)
#   tesseract-ocr + -deu/-nld    → de OCR zelf. LET OP: 'deu' is essentieel bij
#                                  grensmateriaal; met alleen nld+eng werd
#                                  "Straßenabläufe" nog "StraBenablaufe".
#   python3-pil (Pillow)         → roteren van de gerasterde pagina
#   wngerman / wdutch / wamerican → de woordenlijsten voor de poort hierboven
#
# CONFIGURATIE — alles via environment variables (defaults tussen haakjes).
# =============================================================================
import csv, io, os, re, subprocess, sys, tempfile, unicodedata

# Leesrichtingen (graden, tegen de klok in) die geprobeerd worden. Dankzij de
# woordenboek-poort kost een overbodige richting alleen tijd, geen kwaliteit.
# "0 90" volgt de tekenconventie; zet "0 90 180 270" bij twijfel.
ROTATIONS   = [int(x) for x in os.environ.get("OCR_ROTATIONS", "0 90").split()]
# Taalmodellen. Volgorde telt niet; elk extra model kost wat snelheid.
LANGS       = os.environ.get("OCR_LANGS", "deu+nld+eng")
# Page segmentation mode. 3 (automatisch) bleek op de testtekening duidelijk
# beter dan 11 ("sparse text"): 11 vindt meer losse fragmenten, maar bijna
# alleen ruis uit de arceringen.
PSM         = os.environ.get("OCR_PSM", "3")
# Woorden onder deze OCR-betrouwbaarheid vallen af (0-100).
MIN_CONF    = float(os.environ.get("OCR_MIN_CONF", "60"))
# Ondergrens voor het aandeel regels met een herkend woord; daaronder wordt de
# hele leesrichting verworpen. 0,25 zit ruim tussen de gemeten 0,07 en 0,69.
MIN_DICT    = float(os.environ.get("OCR_MIN_DICT_RATIO", "0.25"))
# Rasterresolutie. Hoger dan de scan zelf heeft geen zin (deze scan is 300 dpi).
DPI         = os.environ.get("OCR_DPI", "300")
# Rem op dikke bundels: hooguit zoveel pagina's per document OCR'en.
MAX_PAGES   = int(os.environ.get("OCR_MAX_PAGES", "10"))
# Vangnet tegen geheugenproblemen: grotere pagina's worden eerst verkleind.
MAX_PIXELS  = int(os.environ.get("OCR_MAX_PIXELS", "180000000"))
DICT_FILES  = ["/usr/share/dict/ngerman", "/usr/share/dict/dutch",
               "/usr/share/dict/american-english"]

def norm(w):
    w = unicodedata.normalize("NFKD", w.lower())
    return "".join(c for c in w if c.isalpha())

def load_dict():
    d = set()
    for f in DICT_FILES:
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    n = norm(line.strip())
                    if len(n) >= 4:
                        d.add(n)
        except OSError:
            pass
    return d

WORDS = load_dict()

def lines_from_tsv(tsv):
    rows = list(csv.reader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE))
    if not rows: return []
    I = {k: i for i, k in enumerate(rows[0])}
    if "text" not in I: return []
    grouped = {}
    for r in rows[1:]:
        if len(r) <= I["text"]: continue
        t = r[I["text"]].strip()
        if not t: continue
        try: conf = float(r[I["conf"]])
        except ValueError: continue
        if conf < MIN_CONF: continue
        key = (r[I["block_num"]], r[I["par_num"]], r[I["line_num"]])
        grouped.setdefault(key, []).append(t)
    out = []
    for ws in grouped.values():
        s = re.sub(r"\s+", " ", " ".join(ws)).strip()
        if len(s) >= 3:
            out.append(s)
    return out

def dict_ratio(lines):
    tot = hit = 0
    for line in lines:
        toks = [norm(t) for t in re.findall(r"[^\W\d_]+", line, re.UNICODE)]
        toks = [t for t in toks if len(t) >= 4]
        if not toks: continue
        tot += 1
        if any(t in WORDS for t in toks): hit += 1
    return (hit / tot if tot else 0.0), tot

def ocr_image(img):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        tmp = fh.name
    try:
        img.save(tmp)
        p = subprocess.run(
            ["tesseract", tmp, "-", "-l", LANGS, "--psm", PSM,
             "-c", "preserve_interword_spaces=1", "tsv"],
            capture_output=True, text=True, timeout=int(os.environ.get("OCR_TIMEOUT_SEC", "600")))
        return p.stdout
    finally:
        os.unlink(tmp)

def main(pdf):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdftoppm", "-png", "-gray", "-r", DPI,
                        "-l", str(MAX_PAGES), pdf, os.path.join(d, "pg")],
                       check=True, timeout=int(os.environ.get("OCR_TIMEOUT_SEC", "600")))
        pages = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        kept, seen = [], set()
        for page in pages:
            base = Image.open(os.path.join(d, page)).convert("L")
            if base.width * base.height > MAX_PIXELS:
                scale = (MAX_PIXELS / (base.width * base.height)) ** 0.5
                base = base.resize((int(base.width * scale), int(base.height * scale)))
                print(f"   ↓ {page} verkleind naar {base.size}", file=sys.stderr)
            for deg in ROTATIONS:
                img = base if deg == 0 else base.rotate(deg, expand=True)
                lines = lines_from_tsv(ocr_image(img))
                ratio, n = dict_ratio(lines)
                if n and ratio < MIN_DICT:
                    print(f"   ✗ {page} @{deg}° verworpen "
                          f"(woordherkenning {ratio:.2f} < {MIN_DICT})", file=sys.stderr)
                    continue
                print(f"   ✓ {page} @{deg}° {len(lines)} regels "
                      f"(woordherkenning {ratio:.2f})", file=sys.stderr)
                for line in lines:
                    # Ontdubbelen op de alfanumerieke kern: dezelfde regel uit
                    # twee leesrichtingen verschilt vaak alleen in aangeplakte
                    # lijnresten ("/ > Maßstab: 1:500" vs "Maßstab: 1:500").
                    key = re.sub(r"[^0-9a-z\u00e0-\u00ff]", "", line.lower())
                    if len(key) < 3 or key in seen:
                        continue
                    seen.add(key)
                    kept.append(line)
        print("\n".join(kept))

if __name__ == "__main__":
    main(sys.argv[1])
