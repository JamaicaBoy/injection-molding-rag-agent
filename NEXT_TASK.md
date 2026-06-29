# NEXT_TASK.md

## Current Next Task (2026-06-29)

Publish the fixed `full_release_no_pdf_v1` GitHub Release package:

1. Confirm `github_owner` in `configs/artifact_config.yaml` is the real GitHub owner.
2. Create GitHub Release tag `full-release-v1` and upload `dist/full_release_no_pdf_v1/full_release_no_pdf_v1.zip`.
3. On a clean machine, run `python scripts/download_full_release.py` and confirm automatic SHA256, chunks, and Chroma verification passes.

Do not add the package, extracted artifact, vector store, or source PDFs to the normal Git repository.
