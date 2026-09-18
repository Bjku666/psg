# Repository-local external resources

This directory is reserved for resources downloaded specifically by the
relation-decision qualification line (for example, literature PDFs or a
strong-carrier checkpoint). Nothing is required for the current E0--E4 smoke:
the Fair PSG source is already pinned under `third_party/fair_psg/`, and the
OpenPSG data/checkpoint paths are registered externally in the run contract.

Every future file added here must be accompanied by a small manifest recording
its source URL, download date, SHA256, license, and the experiment that uses
it. Do not silently use a user-global cache.
