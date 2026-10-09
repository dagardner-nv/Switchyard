// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic Messages buffered and streaming codecs.

use std::borrow::Cow;

use serde_json::{Value, json};

use crate::codecs::common::{is_anthropic_request, is_mcp_block, only_fields};
use crate::util::push_lossy;
use crate::{
    ContentBlock, LlmRequest, Message, PRESERVATION_METADATA_KEY, PreservationMetadata, Result,
    Role, ToolChoice, TranslationDiagnostic, TranslationPolicy, WireFormat,
};

mod buffered;
mod stream;

pub use buffered::AnthropicMessagesCodec;
pub use stream::AnthropicMessagesStreamCodec;

pub(super) const ANTHROPIC_TOOLS_KEY: &str = "switchyard_anthropic_tools";

pub(crate) fn prepare_request_tools<'a>(
    request: &'a LlmRequest,
    target: WireFormat,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Cow<'a, LlmRequest>> {
    // Check small definition fields before walking the conversation history.
    let has_provider_tools = (is_anthropic_request(request)
        && (request.extensions.fields.get("mcp_servers")
                .and_then(Value::as_array)
                .is_some_and(|servers| !servers.is_empty())
            || request.extensions.fields.get(ANTHROPIC_TOOLS_KEY)
                .and_then(Value::as_array)
                .is_some_and(|tools| tools.iter().any(is_server_tool))))
        // Preserved source bodies can carry credentials absent from normalized fields.
        || request.preservation.requests.get(&WireFormat::AnthropicMessages.into())
            .is_some_and(|body| {
                body.get("mcp_servers").and_then(Value::as_array)
                    .is_some_and(|servers| !servers.is_empty())
                    || body.get("tools").and_then(Value::as_array)
                        .is_some_and(|tools| tools.iter().any(is_server_tool))
            })
        || request.messages.iter().flat_map(|message| &message.content)
            .chain(request.instructions.iter().flat_map(|instruction| &instruction.content))
            .any(is_provider_tool_content);
    if has_provider_tools {
        push_lossy(
            diagnostics,
            policy,
            format!(
                "Anthropic MCP servers, server tools, and their history are dropped for {target}"
            ),
        )?;
        let mut request = request.clone();
        request.extensions.fields.remove("mcp_servers");
        request.extensions.fields.remove(ANTHROPIC_TOOLS_KEY);
        // Clear exact-replay bodies for every format and the embedded preservation envelope.
        // Any of them could restore dropped tools or forward MCP credentials.
        request.preservation = PreservationMetadata::default();
        if let Some(metadata) = request
            .extensions
            .fields
            .get_mut("metadata")
            .and_then(Value::as_object_mut)
        {
            metadata.remove(PRESERVATION_METADATA_KEY);
        }
        request.instructions.retain_mut(|instruction| {
            drop_provider_tool_content(&mut instruction.content);
            !instruction.content.is_empty()
        });
        request.messages.retain_mut(|message| {
            drop_provider_tool_content(&mut message.content);
            !message.content.is_empty()
        });
        if (request.tools.is_empty() && super::responses::mcp_tools(&request).is_none())
            || matches!(&request.tool_choice, Some(ToolChoice::Tool { name })
                if !request.tools.iter().any(|tool| &tool.name == name))
        {
            request.tool_choice = None;
        }
        return Ok(Cow::Owned(request));
    }
    Ok(Cow::Borrowed(request))
}

fn is_server_tool(tool: &Value) -> bool {
    // Add new server-tool families here; typed client tools must keep their function conversion.
    tool.get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| {
            kind == "mcp_toolset"
                || [
                    "web_search_",
                    "web_fetch_",
                    "code_execution_",
                    "tool_search_tool_",
                    "advisor_",
                ]
                .iter()
                .any(|prefix| kind.starts_with(prefix))
        })
}

fn is_provider_tool_block(block: &Value) -> bool {
    block
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| {
            matches!(kind, "mcp_tool_use" | "server_tool_use") || kind.ends_with("_tool_result")
        })
}

fn is_provider_tool_content(block: &ContentBlock) -> bool {
    match block {
        ContentBlock::Unknown { provider, raw } => {
            provider.as_str() == WireFormat::AnthropicMessages.as_str()
                && is_provider_tool_block(raw)
        }
        ContentBlock::ToolResult(result) => result.content.iter().any(is_provider_tool_content),
        _ => false,
    }
}

fn drop_provider_tool_content(content: &mut Vec<ContentBlock>) {
    content.retain_mut(|block| {
        if let ContentBlock::ToolResult(result) = block {
            drop_provider_tool_content(&mut result.content);
        }
        !is_provider_tool_content(block)
    });
}

pub(super) fn responses_mcp_tool_to_anthropic(tool: &Value) -> Option<(Value, Value)> {
    if !only_fields(
        tool,
        &[
            "type",
            "server_label",
            "server_url",
            "authorization",
            "allowed_tools",
            "require_approval",
        ],
    ) || tool["require_approval"] != "never"
    {
        return None;
    }
    let label = tool["server_label"].as_str()?;
    let url = tool["server_url"].as_str()?;
    if !url.starts_with("https://") {
        return None;
    }
    let mut server = json!({"type": "url", "name": label, "url": url});
    if let Some(token) = tool.get("authorization") {
        server["authorization_token"] = token.clone();
    }
    let mut toolset = json!({"type": "mcp_toolset", "mcp_server_name": label});
    if !tool["allowed_tools"].is_null() {
        let allowed = tool["allowed_tools"].as_array()?;
        let mut configs = serde_json::Map::new();
        for name in allowed {
            configs.insert(name.as_str()?.to_string(), json!({"enabled": true}));
        }
        toolset["default_config"] = json!({"enabled": false});
        toolset["configs"] = Value::Object(configs);
    }
    Some((server, toolset))
}

pub(super) fn responses_mcp_history_to_anthropic(
    messages: &[Message],
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<Message>> {
    let mut translated = Vec::new();
    for message in messages {
        let mut message = message.clone();
        let mut content = Vec::new();
        for block in message.content {
            if !is_mcp_block(&block, WireFormat::OpenAiResponses) {
                content.push(block);
                continue;
            }
            let ContentBlock::Unknown { raw, .. } = block else {
                unreachable!()
            };
            let input = raw["arguments"]
                .as_str()
                .and_then(|args| serde_json::from_str::<Value>(args).ok());
            let is_error = raw["error"]["type"] == "mcp_tool_execution_error";
            let result_content = if is_error {
                raw["error"]["content"].clone()
            } else {
                json!([{"type": "text", "text": raw["output"]}])
            };
            if raw["type"] != "mcp_call"
                || !only_fields(
                    &raw,
                    &[
                        "type",
                        "id",
                        "name",
                        "server_label",
                        "arguments",
                        "output",
                        "error",
                        "status",
                        "approval_request_id",
                    ],
                )
                || (!raw["status"].is_null()
                    && raw["status"] != if is_error { "failed" } else { "completed" })
                || (!raw["error"].is_null() && !is_error)
                || !raw["approval_request_id"].is_null()
                || !input.as_ref().is_some_and(Value::is_object)
                || (!is_error && !raw["output"].is_string())
                || (is_error && !raw["output"].is_null())
                || !(result_content.is_string()
                    || result_content.as_array().is_some_and(|blocks| {
                        blocks
                            .iter()
                            .all(|block| block["type"] == "text" && block["text"].is_string())
                    }))
            {
                push_lossy(
                    diagnostics,
                    policy,
                    "Responses MCP listing, approval, or call data has no Anthropic equivalent; item dropped",
                )?;
                continue;
            }
            message.role = Role::Assistant;
            content.push(ContentBlock::Unknown {provider: WireFormat::AnthropicMessages.into(), raw: json!({
                "type": "mcp_tool_use", "id": raw["id"], "name": raw["name"], "server_name": raw["server_label"], "input": input.unwrap()
            })});
            content.push(ContentBlock::Unknown {
                provider: WireFormat::AnthropicMessages.into(),
                raw: json!({
                    "type": "mcp_tool_result", "tool_use_id": raw["id"], "is_error": is_error,
                    "content": result_content
                }),
            });
        }
        message.content = content;
        if !message.content.is_empty() {
            translated.push(message);
        }
    }
    Ok(translated)
}
