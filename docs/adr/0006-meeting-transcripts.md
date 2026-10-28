# ADR 0006: Meeting transcripts as a first-class source

- Status: accepted
- Date: 2026-10-28

## Context

On a consulting engagement, most of the knowledge is spoken: budgets approved in a steering
committee, a data source dropped during a workshop, a deadline promised in a weekly sync.
Transcripts capture it, but they are long, noisy, often in French and English, and the useful
questions are about *who* said *what* and *when*. Chunking them like wiki pages loses the
speaker, mixes meetings and ignores time.

## Decision

1. **Parsing.** Plain text (`[00:12:34] Name: ...`, `Name (00:12:34): ...`), WebVTT with
   voice tags (Teams, Zoom, Meet) and SRT are parsed into speaker turns. Metadata comes from a
   front matter or `NOTE` block, with the file name (`2026-05-12_steering-committee-1.vtt`) as
   fallback; the meeting type is inferred from the title in French or English.
2. **Cleaning.** Fillers (`um`, `euh`...) and stutters are removed, consecutive turns of the
   same speaker are merged, and emails and phone numbers are masked before indexing
   (`POLYRAG_REDACT_PII`). The phone pattern only accepts numbers starting with `+` or `0`
   and 9 to 15 digits: the first version also masked amounts such as "450 000 euros", which a
   test now guards against.
3. **Chunking by speaker turns.** A turn is never split from its speaker; chunks of about
   1,100 characters overlap by one turn so that a question and its answer stay together. Each
   chunk records the meeting date and type, the participants and the people speaking in it.
4. **Canonical English metadata in the indexed text.** The chunk header includes the meeting
   type in English. Measured on the 15 meeting questions of the golden set, this raised hybrid
   Recall@5 from 0.744 to 0.811: an English question about a "steering committee" now matches a
   French "Comité de pilotage".
5. **Filters resolved by a meeting catalog.** The router may attach a `meeting_filter` (type,
   date range, speakers, latest only) to a meetings step. A catalog built from the chunk
   metadata resolves it into document ids and full speaker names ("Sophie" -> "Sophie
   Bernard"), which every retriever (BM25, Qdrant, in-memory) applies before ranking. An
   over-constrained filter falls back to all meetings rather than hiding everything.
6. **Meeting records in SQL.** An LLM extracts a summary, decisions, action items (owner,
   task, due date) and risks per meeting into the analytics database, so that "which actions
   are due in November?" is a SQL query, not a retrieval problem. Extractions are cached in
   `data/insights/`, keyed by a hash of the transcript and of the extraction prompt version:
   a fresh checkout, the CI and the Docker image get the records without any model call.
7. **Embeddings unchanged.** Two multilingual models were measured against `bge-small-en-v1.5`
   on the same index (MRR@10, dense search):

   | Model | Documents (33 q.) | Meetings (15 q.) | Size |
   |---|---|---|---|
   | `BAAI/bge-small-en-v1.5` | **0.970** | **0.831** | 67 MB |
   | `paraphrase-multilingual-MiniLM-L12-v2` | 0.879 | 0.763 | 220 MB |
   | `paraphrase-multilingual-mpnet-base-v2` | 0.958 | 0.727 | 1 GB |

   The English model wins on both groups: questions, canonical metadata and most documents
   are in English, and names, figures and dates are language-independent. The router adds
   French keywords to meetings queries so that BM25 also matches French transcripts.

## Consequences

- Questions about decisions and commitments get complete answers from SQL and quotes with
  timestamps from the transcripts, in the same answer.
- Extraction quality is measured offline against hand-annotated records
  (`polyrag eval extraction`), so a prompt change that degrades it is visible.
- Speaker attribution relies on the transcript tool's diarization; audio transcription and
  diarization are out of scope.
- Every user sees every meeting. A production deployment would filter retrieval by the
  meetings a user attended (the participants are already in the chunk metadata).
