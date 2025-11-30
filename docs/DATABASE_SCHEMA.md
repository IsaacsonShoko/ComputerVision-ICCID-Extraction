# Database & Storage Schema

This document describes how data is stored and which systems are used.

## Storage Layers
- **S3**: Stores cropped images and diagnostic frames.
  - Path pattern: `s3://{bucket}/run_{run_id}/detection_{detection_id}/crop.jpg`
  - Metadata file: `metadata.json` next to `crop.jpg`

- **Airtable**: Final structured OCR results.
  - Fields:
    - `Run ID` (string)
    - `Detection ID` (string)
    - `ICCID` (string)
    - `Phone Number` (string)
    - `Confidence` (float)
    - `S3 URL` (attachment/URL)
    - `Processed At` (datetime)
    - `Service Provider` (single select)

- **Local logs**: `logs/` folder stores process logs and diagnostic dumps.

## Example S3 object metadata
```json
{
  "run_id": "20251129_abc123",
  "detection_id": "det_001",
  "bbox": {"x":100,"y":150,"w":150,"h":100},
  "confidence": 0.95,
  "ocr_result": null,
  "capture_ts": "2025-11-29T10:30:45Z"
}
```

## Airtable example record
- `Run ID`: 20251129_abc123
- `Detection ID`: det_001
- `ICCID`: 8931041012345678901
- `Phone Number`: 0831234567
- `Confidence`: 0.95
- `S3 URL`: https://s3.eu-north-1.amazonaws.com/ocrstorage4d/run_20251129_abc123/detection_001/crop.jpg
- `Processed At`: 2025-11-29T10:33:00Z

## Retention & Housekeeping
- S3 lifecycle rules recommended: move older than 90 days to Glacier or delete after 180 days.
- Local logs rotated daily via `logrotate` configuration.

## Notes
- Airtable rate limits may apply; batch inserts in n8n are recommended.
- If scaling to high throughput, consider moving to a managed DB (Postgres) for analytics and longer-term retention.
