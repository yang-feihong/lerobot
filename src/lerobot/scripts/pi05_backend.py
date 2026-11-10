"""In-process PI0.5 deployment backend; no HTTP or robot dependencies."""

from __future__ import annotations

from typing import Any, Callable


class Pi05Backend:
    def __init__(self, engine: Any, api: Any, *, input_error: Callable[[str], Exception] = ValueError):
        self.engine = engine
        self.api = api
        self.resources = api.ImageResources()
        self.persistent_http = False
        self.array_transport = None
        self.closed = False
        self.input_error = input_error

    def input_spec(self) -> dict[str, Any]:
        return self.engine.input_spec()

    def submit(self, prepared: Any) -> dict[str, Any]:
        if self.closed:
            raise OSError("Backend session closed")
        with self.resources.lock:
            try:
                packet = self.api.decode_observation_packet(
                    bytes(prepared), prepared_values=prepared,
                    decode_window=True, image_resources=self.resources,
                )
                self.engine.validate_input(packet)
                if self.engine.submit(packet) is False:
                    raise ValueError("Observation is older than the pending input")
            except (ValueError, KeyError, TypeError, OSError, EOFError) as exc:
                raise self.input_error(str(exc)) from exc
            acknowledgement: dict[str, Any] = {}
            if isinstance(packet.model_image_history, self.api.FlatImageHistory):
                history = packet.model_image_history
                self.resources.commit(history.wire_values, history.input_sha256)
                acknowledgement["image_input_sha256"] = history.input_sha256
            return acknowledgement

    def poll(self, after: int, wait_ms: int = 0) -> dict[str, Any] | None:
        if self.closed:
            raise OSError("Backend session closed")
        record = self.engine.latest_after(after, wait_ms)
        return None if record is None else self.api.action_packet_values(record)

    def close(self) -> None:
        self.closed = True
