# switchyard-translation

Pure Rust translation between OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages
request, response, and streaming formats.

The crate translates through provider-neutral LLM types from `switchyard-protocol` and does not
depend on provider SDKs, HTTP servers, Python, or FFI bindings.

MCP definitions and history survive Responses reconstruction, including tool listings and
approval items. Translation between Responses and Anthropic preserves URL servers,
authorization tokens, tool allowlists, and completed calls with text results or tool execution
errors. Responses servers must use `require_approval: "never"` for Anthropic conversion.

MCP data has no Chat Completions equivalent. Settings or history without a target equivalent
follow `LossyConversionPolicy`: reject the conversion, or drop the unsupported data with
diagnostics. This includes Responses approvals and connectors, and Anthropic tool denylists
and deferred loading. Anthropic backends also need the MCP beta header configured as described
in the [Anthropic MCP documentation](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector).

## License

Licensed under the Apache License, Version 2.0. See the
[Switchyard repository](https://github.com/NVIDIA-NeMo/Switchyard) for details.
