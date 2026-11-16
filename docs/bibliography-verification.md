# Publisher-record verification — 16 November 2026

The four records below were checked against publisher sources after OpenReview
forum and API requests returned access challenges. The COLM proceedings site
provides the conference records directly. Its paper pages explicitly state
that BibTeX downloads are unavailable in the pre-proceedings; the 2026 volume
states that DOI assignment is pending. No pages or DOI identifiers were inferred.

| Citation key | Primary evidence | Verified fields and action |
| --- | --- | --- |
| `bereska2024mechanistic` | [TMLR BibTeX](https://jmlr.org/tmlr/papers/bib/ePUVetPKu6.bib), preserved in [`paper/bibliography-records/`](../paper/bibliography-records/bereska2024mechanistic.bib) | Title, two authors, journal, year, ISSN and forum URL. The publisher export uses **Stratis Gavves**. Its month is unspecified; the PDF says August and the journal index says September, so the citation omits a month. |
| `marks2024geometry` | [COLM 2024 proceedings record](https://proceedings.colm.cc/paper_files/paper/2024/hash/32ebb6b560ee58abbdae834e5f37cb5d-Abstract-Conference.html) | Title, Samuel Marks and Max Tegmark, conference year and publication status agree. The citation now links to this accessible publisher record. |
| `teamolmo2025olmo2` | [COLM 2025 proceedings record](https://proceedings.colm.cc/paper_files/paper/2025/hash/e68f3154ffebe9acddb8047cc197cb71-Abstract-Conference.html), [publisher-hosted PDF](https://openreview.net/pdf/ee2c137da42a7d7cd97b58127c3b38b1bd47107d.pdf) | Title, conference year and status agree. The indexed PDF's byline begins **OLMo Team**, followed by its members; the citation retains that corporate author. The conference name is now “Second Conference on Language Modeling.” |
| `fu2026convergent` | [COLM 2026 proceedings record](https://proceedings.colm.cc/paper_files/paper/2026/hash/7d6e43c0800b2465e4e3cb1012e6820e-Abstract-Conference.html), [accepted-paper list](https://colm.eventhosts.cc/Conferences/2026/AcceptedPapers) | Title, all five authors, conference year and publication status agree. The citation uses “Third Conference on Language Modeling” and the proceedings URL, without the preprint DOI. |

The publisher record uses the full “Large Language Model” wording in the
Marks–Tegmark title; the PDF abbreviates it to “LLM.” The citation follows the
publisher record. Protected capitalization for AI, OLMo and COLM is preserved.

The HTML metadata dates the COLM website records to 15 October 2026, including
the 2024 and 2025 volumes. Citation years follow the conference volume and
accepted-paper records. The published Fu PDF was downloaded and its byline and
COLM 2026 header checked; its hash is in the verification record.

The available publisher fields are verified. The raw OpenReview exports could
not be retrieved, and COLM does not offer replacement exports on these pages.
The OLMo corporate byline was checked in the indexed publisher PDF; a fresh PDF
download encountered the same OpenReview challenge, and the replacement
proceedings PDF URL returned HTTP 404. These access limits do not
establish a metadata error or justify replacing a documented author byline.

The accompanying `paper/bibliography-records/verification.json` records source
URLs, saved publisher-record hashes and citation-entry hashes.
