"""Verifikasi metadata rujukan [19] sampai [36] dan [42] pada naskah.

Tujuan
------
Memeriksa metadata bibliografi yang ditulis pada naskah BAB II skripsi
bagian "Rujukan Teori Bab II" terhadap sumber otoritatif daring, yaitu Crossref
REST API (untuk entri ber-DOI dan pencarian bibliografis) dan arXiv API (untuk
entri pracetak). Keluarannya dipakai untuk memutuskan entri mana yang perlu
dikoreksi sebelum bab masuk naskah final.

Sumber yang dipakai: Crossref REST API (DOI dan pencarian bibliografis),
arXiv API (pracetak), dan OpenAlex API (entri lama atau prosiding yang tidak
terindeks Crossref, misalnya terbitan pra-DOI dan ACL Anthology).

Masukan
-------
Tidak ada berkas masukan. Metadata yang diklaim ditulis sebagai konstanta
`CLAIMS` pada modul ini, disalin apa adanya dari naskah BAB II.

Keluaran
--------
1. Ringkasan per rujukan ke stdout.
2. Berkas JSON berisi respons mentah dan hasil perbandingan:
   `_01_python-scripts/outputs/verify_references_report.json`

Perintah jalankan (PowerShell, dari akar folder skripsi)
--------------------------------------------------------
    .\\skripsi-venv\\Scripts\\python.exe .\\_01_python-scripts\\verify_references.py

Catatan: hanya memakai pustaka standar ditambah `certifi` untuk berkas
sertifikat root, karena penyimpanan sertifikat Windows pada mesin ini tidak
memuat penerbit perantara yang dipakai arXiv.
"""

from __future__ import annotations

import json
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import certifi

# --------------------------------------------------------------------------- #
# Konstanta
# --------------------------------------------------------------------------- #

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "verify_references_report.json"

CROSSREF_WORK_URL = "https://api.crossref.org/works/{doi}"
CROSSREF_QUERY_URL = "https://api.crossref.org/works"
ARXIV_QUERY_URL = "https://export.arxiv.org/api/query"
OPENALEX_QUERY_URL = "https://api.openalex.org/works"

SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# Crossref meminta identifikasi pemanggil lewat User-Agent berisi surel kontak.
USER_AGENT = "skripsi-ref-verifier/1.0 (mailto:noreply@example.com)"
REQUEST_TIMEOUT_S = 30
POLITE_DELAY_S = 1.0

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class Claim:
    """Metadata satu rujukan sebagaimana tertulis di dokumen."""

    ref: str
    title: str
    authors: str
    container: str
    year: int
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    search_title: str | None = None
    notes: str = ""
    manual_source: str = ""
    """Bila diisi, entri dinyatakan terverifikasi manual terhadap laman resmi
    yang disebutkan, karena API otomatis tidak dapat memastikannya. Selisih
    yang dilaporkan API tetap disimpan pada laporan JSON sebagai catatan."""

    def query_title(self) -> str:
        return self.search_title or self.title


CLAIMS: list[Claim] = [
    Claim(
        ref="19",
        title="Attention Is All You Need",
        authors="A. Vaswani et al.",
        container="Advances in Neural Information Processing Systems 30 (NeurIPS 2017)",
        year=2017,
        arxiv_id="1706.03762",
        manual_source=(
            "papers.nips.cc laman NIPS 2017: judul dan daftar penulis cocok, "
            "volume 30. OpenAlex melaporkan tahun 2025 (cacat indeks, diabaikan). "
            "Laman resmi tidak mencantumkan nomor halaman."
        ),
    ),
    Claim(
        ref="20",
        title=(
            "Connectionist temporal classification: Labelling unsegmented "
            "sequence data with recurrent neural networks"
        ),
        authors="A. Graves, S. Fernandez, F. Gomez, and J. Schmidhuber",
        container="Proceedings of the 23rd International Conference on Machine Learning (ICML)",
        year=2006,
        pages="369-376",
        doi="10.1145/1143844.1143891",
    ),
    Claim(
        ref="21",
        title=(
            "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech "
            "Representations"
        ),
        authors="A. Baevski, H. Zhou, A. Mohamed, and M. Auli",
        container="Advances in Neural Information Processing Systems 33 (NeurIPS 2020)",
        year=2020,
        arxiv_id="2006.11477",
    ),
    Claim(
        ref="22",
        title="Unsupervised Cross-lingual Representation Learning for Speech Recognition",
        authors="A. Conneau, A. Baevski, R. Collobert, A. Mohamed, and M. Auli",
        container="Interspeech 2021, ISCA",
        year=2021,
        pages="2426-2430",
        doi="10.21437/Interspeech.2021-329",
        arxiv_id="2006.13979",
    ),
    Claim(
        ref="23",
        title="Binary codes capable of correcting deletions, insertions, and reversals",
        authors="V. I. Levenshtein",
        container="Soviet Physics Doklady",
        year=1966,
        volume="10",
        issue="8",
        pages="707-710",
        notes="Terbitan 1966 pra-DOI, tidak terindeks Crossref maupun OpenAlex.",
        manual_source=(
            "NASA ADS bibcode 1966SPhD...10..707L: vol. 10, no. 8, hlm. 707-710, "
            "1966. Terjemahan dari Doklady Akademii Nauk SSSR vol. 163, no. 4, "
            "hlm. 845-848, 1965."
        ),
    ),
    Claim(
        ref="24",
        title=(
            "From WER and RIL to MER and WIL: improved evaluation measures for "
            "connected speech recognition"
        ),
        authors="A. C. Morris, V. Maier, and P. Green",
        container="Interspeech 2004 (ICSLP), ISCA",
        year=2004,
        pages="2765-2768",
        doi="10.21437/Interspeech.2004-668",
    ),
    Claim(
        ref="25",
        title="Bootstrap Methods: Another Look at the Jackknife",
        authors="B. Efron",
        container="The Annals of Statistics",
        year=1979,
        volume="7",
        issue="1",
        pages="1-26",
        doi="10.1214/aos/1176344552",
        manual_source=(
            "Crossref tidak memuat medan halaman untuk record Project Euclid ini. "
            "Rentang 1-26 dikonfirmasi lewat metadata JSTOR 2958830 dan MR0515681."
        ),
    ),
    Claim(
        ref="26",
        title="Statistical Significance Tests for Machine Translation Evaluation",
        authors="P. Koehn",
        container=(
            "Proceedings of the 2004 Conference on Empirical Methods in Natural "
            "Language Processing (EMNLP)"
        ),
        year=2004,
        pages="388-395",
        manual_source=(
            "ACL Anthology W04-3250: hlm. 388-395, Barcelona, Spain, penerbit ACL, "
            "medan DOI kosong pada laman resmi."
        ),
    ),
    Claim(
        ref="27",
        title="A Simple Sequentially Rejective Multiple Test Procedure",
        authors="S. Holm",
        container="Scandinavian Journal of Statistics",
        year=1979,
        volume="6",
        issue="2",
        pages="65-70",
        manual_source=(
            "OpenAlex mengonfirmasi vol. 6, hlm. 65-70, 1979, tetapi tidak memuat "
            "nomor terbitan. Nomor 2 dikonfirmasi lewat metadata JSTOR 4615733 "
            "dan MR0538597."
        ),
    ),
    Claim(
        ref="28",
        title="A New Look at the Statistical Model Identification",
        authors="H. Akaike",
        container="IEEE Transactions on Automatic Control",
        year=1974,
        volume="19",
        issue="6",
        pages="716-723",
        doi="10.1109/TAC.1974.1100705",
    ),
    Claim(
        ref="29",
        title="KenLM: Faster and Smaller Language Model Queries",
        authors="K. Heafield",
        container="Proceedings of the Sixth Workshop on Statistical Machine Translation (WMT)",
        year=2011,
        pages="187-197",
        manual_source=(
            "ACL Anthology W11-2123: hlm. 187-197, Edinburgh, Scotland, penerbit "
            "ACL, medan DOI kosong pada laman resmi."
        ),
    ),
    Claim(
        ref="30",
        title="LoRA: Low-Rank Adaptation of Large Language Models",
        authors="E. J. Hu et al.",
        container="International Conference on Learning Representations (ICLR)",
        year=2022,
        arxiv_id="2106.09685",
        manual_source=(
            "OpenReview forum nZeVKeeFYf9 dan iclr.cc/virtual/2022/poster/6319: "
            "terbit sebagai poster ICLR 2022. Pracetak arXiv 2106.09685 v1 "
            "17 Juni 2021, v2 16 Oktober 2021, tanpa medan journal-ref."
        ),
    ),
    Claim(
        ref="31",
        title=(
            "Deep Neural Networks for Acoustic Modeling in Speech Recognition: "
            "The Shared Views of Four Research Groups"
        ),
        authors="G. Hinton et al.",
        container="IEEE Signal Processing Magazine",
        year=2012,
        volume="29",
        issue="6",
        pages="82-97",
        doi="10.1109/MSP.2012.2205597",
    ),
    Claim(
        ref="32",
        title="CRISP-DM: Towards a Standard Process Model for Data Mining",
        authors="R. Wirth and J. Hipp",
        container=(
            "Proceedings of the 4th International Conference on the Practical "
            "Applications of Knowledge Discovery and Data Mining (PADD)"
        ),
        year=2000,
        pages="29-39",
        notes="Kerangka proses penelitian pada naskah BAB III.",
        manual_source=(
            "Tidak terindeks Crossref maupun OpenAlex, dan prosidingnya tidak "
            "memiliki DOI. Judul, penulis, tahun, dan rentang halaman 29-39 "
            "dikonfirmasi lewat rekaman Semantic Scholar CorpusID 1211505 dan "
            "berkas prosiding yang beredar luas. Venue: Proceedings of the 4th "
            "International Conference on the Practical Applications of Knowledge "
            "Discovery and Data Mining, London, 2000. Sebagian literatur menulis "
            "tahun 1999, yaitu tahun pengembangan model, bukan tahun terbit "
            "makalah ini."
        ),
    ),
    Claim(
        ref="33",
        title="A Systematic Literature Review on Applying CRISP-DM Process Model",
        authors="C. Schroer, F. Kruse, and J. M. Gomez",
        container="Procedia Computer Science",
        year=2021,
        volume="181",
        pages="526-534",
        doi="10.1016/j.procs.2021.01.199",
        notes=(
            "Dasar bahwa CRISP-DM lazim diadaptasi lintas domain penelitian, "
            "dipakai pada naskah BAB III Subbab 3.2."
        ),
    ),
    Claim(
        ref="34",
        title="CRISP-DM 1.0: Step-by-step data mining guide",
        authors=(
            "P. Chapman, J. Clinton, R. Kerber, T. Khabaza, T. Reinartz, "
            "C. Shearer, and R. Wirth"
        ),
        container="CRISP-DM Consortium, Laporan Teknis CRISPWP-0800",
        year=2000,
        notes=(
            "Sumber primer definisi keenam fase CRISP-DM, dipakai pada naskah "
            "BAB III Subbab 3.2.1. Berbeda dari [32], yang merupakan makalah "
            "konferensi tentang CRISP-DM dan bukan panduan resminya."
        ),
        manual_source=(
            "Laporan teknis konsorsium, bukan terbitan berkala, sehingga tidak "
            "ber-DOI dan tidak terindeks Crossref maupun OpenAlex. Daftar tujuh "
            "penulis, judul, nomor dokumen CRISPWP-0800, dan tahun 2000 "
            "dipastikan langsung dari halaman judul dan halaman hak cipta berkas "
            "PDF aslinya. Judul aslinya memakai sentence case, yaitu "
            "'Step-by-step data mining guide', bukan Title Case seperti pada "
            "banyak sitasi sekunder. Kata pengantar bertanggal Agustus 2000, "
            "sedangkan halaman hak cipta menulis '1999, 2000', sehingga sebagian "
            "literatur salah menulis tahunnya sebagai 1999. Tautan arsip: "
            "http://web.archive.org/web/20101231174706/"
            "http://www.crisp-dm.org/CRISPWP-0800.pdf (domain resmi crisp-dm.org "
            "sudah tidak aktif)."
        ),
    ),
    Claim(
        ref="35",
        title=(
            "Towards CRISP-ML(Q): A Machine Learning Process Model with Quality "
            "Assurance Methodology"
        ),
        authors=(
            "S. Studer, T. B. Bui, C. Drescher, A. Hanuschkin, L. Winkler, "
            "S. Peters, and K.-R. Muller"
        ),
        container="Machine Learning and Knowledge Extraction",
        year=2021,
        volume="3",
        issue="2",
        pages="392-413",
        doi="10.3390/make3020020",
        notes=(
            "Dasar bahwa model proses baku untuk machine learning belum ada "
            "sehingga CRISP-DM perlu disesuaikan. Dipakai pada naskah BAB III "
            "Subbab 3.2.1. Hindari DOI pracetak "
            "10.20944/preprints202103.0135.v1; yang dipakai versi jurnal."
        ),
    ),
    Claim(
        ref="36",
        title=(
            "CRISP-DM Twenty Years Later: From Data Mining Processes to Data "
            "Science Trajectories"
        ),
        authors=(
            "F. Martinez-Plumed, L. Contreras-Ochando, C. Ferri, "
            "J. Hernandez-Orallo, M. Kull, N. Lachiche, M. J. Ramirez-Quintana, "
            "and P. Flach"
        ),
        container="IEEE Transactions on Knowledge and Data Engineering",
        year=2021,
        volume="33",
        issue="8",
        pages="3048-3061",
        doi="10.1109/TKDE.2019.2962680",
        notes=(
            "Dasar bahwa CRISP-DM tetap berlaku bagi proyek bertujuan "
            "terdefinisi, dan perlu lebih fleksibel bila eksploratif. Dipakai "
            "pada naskah BAB III Subbab 3.2.1. Nomor DOI berkode 2019 karena "
            "early access, sedangkan isu terbitnya Agustus 2021."
        ),
    ),
    Claim(
        ref="42",
        title="From Data Mining to Knowledge Discovery in Databases",
        authors="U. Fayyad, G. Piatetsky-Shapiro, and P. Smyth",
        container="AI Magazine",
        year=1996,
        volume="17",
        issue="3",
        pages="37-54",
        doi="10.1609/aimag.v17i3.1230",
        notes=(
            "Sumber primer proses KDD, kerangka proses penelitian pada naskah "
            "BAB III Subbab 3.2.1. Berbeda dari bab buku 'From Data Mining to "
            "Knowledge Discovery: An Overview' karya penulis yang sama pada "
            "Advances in Knowledge Discovery and Data Mining (AAAI/MIT Press, "
            "1996), dan dari artikel Communications of the ACM 39(11) 1996."
        ),
        manual_source=(
            "doi.org/ra menyatakan DOI terdaftar di Crossref dan doi.org "
            "meneruskannya ke laman artikel AAAI OJS, tetapi Crossref REST API "
            "mengembalikan HTTP 404 dan OpenAlex tidak memuat rekaman ber-DOI "
            "(diperiksa 16 Oktober 2026). Judul, ketiga penulis, vol. 17, no. 3, "
            "dan tahun 1996 dipastikan dari laman artikel "
            "ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/1230 "
            "dan berkas PDF resminya, yang bermetadata 'AI Magazine Volume 17 "
            "Number 3 (Fall 1996)', berisi 18 halaman, dan bernomor halaman "
            "tercetak 37 sampai 54. Laman OJS menulis tanggal terbit 15 Maret "
            "1996, berlawanan dengan label terbitan Fall 1996, sehingga yang "
            "dicantumkan hanya tahunnya."
        ),
    ),
]


# --------------------------------------------------------------------------- #
# Utilitas jaringan
# --------------------------------------------------------------------------- #


def http_get(url: str, params: dict[str, Any] | None = None) -> bytes:
    """Ambil isi URL sebagai bytes, dengan User-Agent sopan."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(
        request, timeout=REQUEST_TIMEOUT_S, context=SSL_CONTEXT
    ) as response:
        return response.read()


def fetch_crossref_by_doi(doi: str) -> dict[str, Any] | None:
    try:
        raw = http_get(CROSSREF_WORK_URL.format(doi=urllib.parse.quote(doi)))
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}"}
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"error": f"jaringan: {exc}"}
    return json.loads(raw).get("message")


def fetch_crossref_by_title(title: str, rows: int = 3) -> list[dict[str, Any]]:
    params = {"query.bibliographic": title, "rows": rows}
    try:
        raw = http_get(CROSSREF_QUERY_URL, params)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        return [{"error": f"jaringan: {exc}"}]
    return json.loads(raw).get("message", {}).get("items", [])


def fetch_openalex_by_title(title: str, rows: int = 5) -> list[dict[str, Any]]:
    """Cari di OpenAlex. Dipakai untuk entri pra-DOI dan prosiding ACL."""
    params = {
        "filter": f"title.search:{title}",
        "per-page": rows,
        "mailto": "noreply@example.com",
    }
    try:
        raw = http_get(OPENALEX_QUERY_URL, params)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        return [{"error": f"jaringan: {exc}"}]
    return [openalex_summary(item) for item in json.loads(raw).get("results", [])]


def openalex_summary(work: dict[str, Any]) -> dict[str, Any]:
    """Ambil medan yang relevan dari satu record OpenAlex."""
    biblio = work.get("biblio") or {}
    first, last = biblio.get("first_page"), biblio.get("last_page")
    location = (work.get("primary_location") or {}).get("source") or {}
    return {
        "title": normalise_space(work.get("title") or work.get("display_name") or ""),
        "container": normalise_space(location.get("display_name") or ""),
        "authors": [
            normalise_space((a.get("author") or {}).get("display_name") or "")
            for a in work.get("authorships", [])
        ],
        "year": work.get("publication_year"),
        "volume": biblio.get("volume"),
        "issue": biblio.get("issue"),
        "pages": f"{first}-{last}" if first and last else first or last,
        "doi": (work.get("doi") or "").replace("https://doi.org/", "") or None,
        "type": work.get("type"),
        "openalex_id": work.get("id"),
    }


def fetch_arxiv(arxiv_id: str) -> dict[str, Any]:
    params = {"id_list": arxiv_id, "max_results": 1}
    try:
        raw = http_get(ARXIV_QUERY_URL, params)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        return {"error": f"jaringan: {exc}"}

    root = ET.fromstring(raw)
    entry = root.find("atom:entry", ARXIV_NS)
    if entry is None:
        return {"error": "entri tidak ditemukan"}

    def text(tag: str) -> str:
        node = entry.find(f"atom:{tag}", ARXIV_NS)
        return normalise_space(node.text or "") if node is not None else ""

    authors = [
        normalise_space(node.findtext("atom:name", default="", namespaces=ARXIV_NS))
        for node in entry.findall("atom:author", ARXIV_NS)
    ]
    return {
        "id": text("id"),
        "title": text("title"),
        "published": text("published"),
        "updated": text("updated"),
        "authors": authors,
        "comment": normalise_space(
            entry.findtext(
                "{http://arxiv.org/schemas/atom}comment", default="", namespaces={}
            )
            or ""
        ),
        "journal_ref": normalise_space(
            entry.findtext(
                "{http://arxiv.org/schemas/atom}journal_ref", default="", namespaces={}
            )
            or ""
        ),
        "doi": normalise_space(
            entry.findtext(
                "{http://arxiv.org/schemas/atom}doi", default="", namespaces={}
            )
            or ""
        ),
    }


# --------------------------------------------------------------------------- #
# Utilitas perbandingan
# --------------------------------------------------------------------------- #


def normalise_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalise_title(value: str) -> str:
    """Turunkan judul ke bentuk banding: huruf kecil, hanya alfanumerik."""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def normalise_pages(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"[^0-9]+", "-", value).strip("-")


def crossref_summary(message: dict[str, Any]) -> dict[str, Any]:
    """Ambil medan yang relevan dari satu record Crossref."""
    if "error" in message:
        return message

    titles = message.get("title") or [""]
    containers = message.get("container-title") or [""]
    authors = [
        normalise_space(f"{a.get('given', '')} {a.get('family', '')}")
        for a in message.get("author", [])
    ]
    issued = message.get("issued", {}).get("date-parts", [[None]])[0]
    return {
        "title": normalise_space(titles[0]),
        "container": normalise_space(containers[0]) if containers else "",
        "authors": authors,
        "year": issued[0] if issued else None,
        "volume": message.get("volume"),
        "issue": message.get("issue"),
        "pages": message.get("page"),
        "doi": message.get("DOI"),
        "type": message.get("type"),
        "publisher": message.get("publisher"),
    }


def compare(claim: Claim, found: dict[str, Any]) -> list[str]:
    """Bandingkan klaim dokumen terhadap metadata temuan. Kembalikan selisih."""
    issues: list[str] = []
    if "error" in found:
        return [f"sumber gagal diambil: {found['error']}"]

    claimed_title = normalise_title(claim.query_title())
    found_title = normalise_title(found.get("title", ""))
    if claimed_title and found_title and claimed_title != found_title:
        if claimed_title not in found_title and found_title not in claimed_title:
            issues.append(f"judul beda: dokumen='{claim.title}' sumber='{found['title']}'")

    if claim.year and found.get("year") and claim.year != found["year"]:
        issues.append(f"tahun beda: dokumen={claim.year} sumber={found['year']}")

    if claim.volume and found.get("volume") and str(claim.volume) != str(found["volume"]):
        issues.append(f"volume beda: dokumen={claim.volume} sumber={found['volume']}")

    if claim.issue and found.get("issue") and str(claim.issue) != str(found["issue"]):
        issues.append(f"nomor beda: dokumen={claim.issue} sumber={found['issue']}")

    claimed_pages = normalise_pages(claim.pages)
    found_pages = normalise_pages(found.get("pages"))
    if claimed_pages and found_pages and claimed_pages != found_pages:
        issues.append(f"halaman beda: dokumen={claim.pages} sumber={found['pages']}")

    if claim.doi and found.get("doi") and claim.doi.lower() != str(found["doi"]).lower():
        issues.append(f"DOI beda: dokumen={claim.doi} sumber={found['doi']}")

    return issues


# --------------------------------------------------------------------------- #
# Alur utama
# --------------------------------------------------------------------------- #


@dataclass
class Result:
    ref: str
    claimed_title: str
    sources: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    manual_source: str = ""
    status: str = "BELUM"


def verify(claim: Claim) -> Result:
    result = Result(ref=claim.ref, claimed_title=claim.title)

    if claim.doi:
        message = fetch_crossref_by_doi(claim.doi)
        summary = crossref_summary(message or {"error": "kosong"})
        result.sources["crossref_doi"] = summary
        result.issues.extend(compare(claim, summary))
        time.sleep(POLITE_DELAY_S)

    if claim.arxiv_id:
        arxiv = fetch_arxiv(claim.arxiv_id)
        result.sources["arxiv"] = arxiv
        if "error" not in arxiv:
            arxiv_as_record = {
                "title": arxiv["title"],
                "year": int(arxiv["published"][:4]) if arxiv.get("published") else None,
            }
            # Tahun arXiv adalah tahun pracetak, bukan tahun venue; jangan diadu.
            arxiv_as_record.pop("year")
            result.issues.extend(compare(claim, arxiv_as_record))
        else:
            result.issues.append(f"arXiv gagal: {arxiv['error']}")
        time.sleep(POLITE_DELAY_S)

    if not claim.doi:
        candidates = [crossref_summary(item) for item in fetch_crossref_by_title(claim.query_title())]
        result.sources["crossref_search"] = candidates
        target = normalise_title(claim.query_title())
        match = next(
            (c for c in candidates if "error" not in c and normalise_title(c.get("title", "")) == target),
            None,
        )
        if match:
            result.sources["crossref_match"] = match
            result.issues.extend(compare(claim, match))
        else:
            result.issues.append("tidak ada padanan judul persis pada pencarian Crossref")
        time.sleep(POLITE_DELAY_S)

        # Crossref tidak memuat terbitan pra-DOI dan sebagian prosiding ACL.
        # OpenAlex dipakai sebagai sumber kedua khusus untuk kasus itu.
        if not match:
            candidates = fetch_openalex_by_title(claim.query_title())
            result.sources["openalex_search"] = candidates
            target = normalise_title(claim.query_title())
            oa_match = next(
                (
                    c
                    for c in candidates
                    if "error" not in c and normalise_title(c.get("title", "")) == target
                ),
                None,
            )
            if oa_match:
                result.sources["openalex_match"] = oa_match
                result.issues = [
                    i
                    for i in result.issues
                    if i != "tidak ada padanan judul persis pada pencarian Crossref"
                ]
                result.issues.extend(compare(claim, oa_match))
            else:
                result.issues.append("tidak ada padanan judul persis pada OpenAlex")
            time.sleep(POLITE_DELAY_S)

    if claim.manual_source:
        result.manual_source = claim.manual_source
        # API otomatis tidak dapat memastikan entri ini. Selisih yang dilaporkan
        # API diturunkan menjadi catatan, bukan temuan, karena sudah dijawab oleh
        # laman resmi yang disebut pada `manual_source`.
        result.notes = result.issues
        result.issues = []
        result.status = "SESUAI (manual)"
    else:
        result.status = "SESUAI" if not result.issues else "PERLU TINJAU"
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[Result] = []
    for claim in CLAIMS:
        print(f"[{claim.ref}] memeriksa: {claim.title[:60]}...", flush=True)
        results.append(verify(claim))

    print("\n" + "=" * 78)
    print("RINGKASAN")
    print("=" * 78)
    for result in results:
        print(f"\n[{result.ref}] {result.status}  {result.claimed_title[:55]}")
        for issue in result.issues:
            print(f"    ! {issue}")
        if result.manual_source:
            print(f"    verifikasi manual: {result.manual_source}")
        for note in result.notes:
            print(f"    catatan API: {note}")
        for key, value in result.sources.items():
            if (
                key in {"crossref_doi", "crossref_match", "openalex_match"}
                and isinstance(value, dict)
                and "error" not in value
            ):
                print(
                    f"    sumber {key}: {value.get('container')} | vol {value.get('volume')} "
                    f"| no {value.get('issue')} | hlm {value.get('pages')} | {value.get('year')}"
                )
            if key == "arxiv" and isinstance(value, dict) and "error" not in value:
                print(
                    f"    sumber arxiv: {value.get('id')} | terbit {value.get('published')[:10]} "
                    f"| journal_ref='{value.get('journal_ref')}' | doi='{value.get('doi')}'"
                )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_document": "_02_context-MD-files/bab-2.md (baris 616-640)",
        "results": [
            {
                "ref": r.ref,
                "claimed_title": r.claimed_title,
                "status": r.status,
                "issues": r.issues,
                "notes": r.notes,
                "manual_source": r.manual_source,
                "sources": r.sources,
            }
            for r in results
        ],
    }
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nLaporan lengkap: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
