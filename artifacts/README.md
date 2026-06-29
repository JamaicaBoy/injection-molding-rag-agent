# Release artifacts

The public full-corpus Demo uses one fixed data package: `full_release_no_pdf_v1`.
It contains chunks, metadata, knowledge cards, and the Chroma vector store, but no
paper PDFs.

Download and verify it with:

```bash
python scripts/download_full_release.py
```

The extracted package is installed at `artifacts/full_release_no_pdf_v1/` and is
excluded from the normal Git repository.
