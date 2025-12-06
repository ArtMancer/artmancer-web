"""Service layer for business logic."""

# Lazy import to avoid loading heavy dependencies in CPU-only services
# GenerationService is only imported when actually needed
__all__ = []


def get_generation_service():
    """Lazy import of GenerationService."""
    from .generation_service import GenerationService
    return GenerationService
