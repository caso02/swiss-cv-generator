# Old/Deprecated Files

Dieser Ordner enthält veraltete, doppelte oder nicht mehr verwendete Dateien aus dem Swiss CV Generator Projekt.

## Inhalt

- **Alte CLI-Implementierungen**: `cli.py`, `mvp_cli.py`, `run.py`
- **Veraltete Generatoren**: `generators/`, `swisscv/`, `swiss_cv_generator/`
- **Alte Export-Funktionen**: `exporter.py`, `exporters/`
- **Veraltete Scripts**: Verschiedene alte Batch/Generate Scripts
- **Alte Data Files**: CSV/JSON Dateien (MongoDB ist jetzt die Quelle)
- **Cache/Temp**: `cache/`, `out/`, `swiss_tech_cvs/`

## Status

Diese Dateien wurden hierher verschoben, um das Projekt aufzuräumen.
Sie können gelöscht werden, wenn sichergestellt ist, dass sie nicht mehr benötigt werden.

**Verschoben am**: $(date)
**Anzahl Dateien**: $(find . -type f | wc -l)
**Größe**: $(du -sh . | cut -f1)
