class UpstreamError(Exception):
    """A dependency (LLM or Superhero API) failed or timed out."""

    def __init__(self, service: str, detail: str, timeout: bool = False):
        super().__init__(f"{service}: {detail}")
        self.service = service
        self.detail = detail
        self.timeout = timeout
