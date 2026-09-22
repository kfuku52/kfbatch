"""Shared provenance and completeness metadata for parsed scheduler tables."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ParseQuality:
    candidate_rows: int = 0
    recognized_rows: int = 0
    rejected_rows: int = 0
    missing_fields: tuple[str, ...] = ()
    scope: str = "observed"

    @property
    def complete(self):
        return self.rejected_rows == 0 and not self.missing_fields


def attach_parse_quality(frame, *, candidate_rows, missing_fields=(), scope="observed"):
    quality = ParseQuality(
        candidate_rows=candidate_rows,
        recognized_rows=len(frame),
        rejected_rows=max(candidate_rows - len(frame), 0),
        missing_fields=tuple(missing_fields),
        scope=scope,
    )
    # Keep legacy attributes and a serializable, provider-independent description.
    frame.attrs.update(asdict(quality))
    frame.attrs["parse_quality"] = asdict(quality)
    return frame


def rejected_rows(frame):
    return int(frame.attrs.get("rejected_rows", 0))
