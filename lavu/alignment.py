"""Guard the historical zero fallback; explicit and nonzero offsets are retained."""
import math

def resolve_lip_offset(explicit_offset, metadata_offset, lip_frames, tongue_frames):
    """Preserve the recorded procedure; guard a collapsed zero-duration fallback.

    Excess lip length cannot recover a positive onset if the lip trajectory is
    shorter than the tongue trajectory. Restore the metadata onset in that case.
    Nonzero duration-based offsets are retained for this retrospective correction;
    discrepancies are audited separately. This rule does not use recognition errors.
    """
    if explicit_offset is not None:
        return int(explicit_offset)
    fallback=max(0,int(lip_frames)-int(tongue_frames))
    if metadata_offset is None:
        return fallback
    value=float(metadata_offset)
    if not math.isfinite(value) or value<0 or value>=lip_frames:
        raise ValueError('Invalid metadata-derived stream onset')
    onset=int(round(value))
    return onset if fallback==0 and onset>0 else fallback
