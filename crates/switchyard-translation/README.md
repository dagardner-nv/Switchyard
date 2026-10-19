# switchyard-translation

Pure Rust translation between OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages
request, response, and streaming formats.

The crate translates through provider-neutral LLM types from `switchyard-protocol` and does not
depend on provider SDKs, HTTP servers, Python, or FFI bindings.

## Stream termination

The `encode_stream` helpers return provider error events as `LlmStreamError::Upstream`
and stop the stream. Errors detected during finalization use the same error channel.
Content emitted before the error is preserved.

Responses `response.incomplete` events recover missing text, reasoning, function calls,
and usage from the final snapshot without repeating streamed content. The translated
stream retains its length-limit stop reason.

## License

Licensed under the Apache License, Version 2.0. See the
[Switchyard repository](https://github.com/NVIDIA-NeMo/Switchyard) for details.
