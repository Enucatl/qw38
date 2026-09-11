"""Fail-closed errors for the OPT-083 quality framework."""


class QualityFrameworkError(AssertionError):
    """Inadmissible quality-framework evidence or a fail-closed gate."""
