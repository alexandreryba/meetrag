# Meeting-record extraction

10 meetings, hand-annotated gold records, extraction model `claude-opus-5-5`. Action items match when the owner is the same and the tasks have a cosine similarity of at least 0.7.

| Metric | Value |
|---|---|
| Action items: precision | 100.0% (25/25) |
| Action items: recall | 100.0% (25/25) |
| Action items: F1 | 100.0% |
| Due dates correct (matched items) | 96.0% |
| Decisions: recall | 100.0% (21 extracted for 16 expected) |
