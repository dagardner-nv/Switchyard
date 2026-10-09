// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI Responses buffered and streaming codecs.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use serde_json::{Value, json};

use crate::codecs::anthropic::{
    ANTHROPIC_TOOLS_KEY, responses_mcp_history_to_anthropic, responses_mcp_tool_to_anthropic,
};
use crate::codecs::common::{
    RESPONSES_MCP_TOOLS_KEY, is_anthropic_request, is_mcp_block, only_fields,
};
use crate::util::push_lossy;
use crate::{
    ContentBlock, LlmRequest, Message, PRESERVATION_METADATA_KEY, PreservationMetadata, Result,
    Role, ToolChoice, TranslationDiagnostic, TranslationPolicy, WireFormat,
};

mod buffered;
mod stream;

pub use buffered::OpenAiResponsesCodec;
pub use stream::OpenAiResponsesStreamCodec;

pub(crate) fn mcp_tools(request: &LlmRequest) -> Option<&Vec<Value>> {
    request
        .extensions
        .fields
        .get(RESPONSES_MCP_TOOLS_KEY)
        .and_then(Value::as_array)
        .filter(|tools| !tools.is_empty())
}

pub(crate) fn is_mcp_item(item: &Value) -> bool {
    matches!(
        item["type"].as_str(),
        Some("mcp_list_tools" | "mcp_call" | "mcp_approval_request" | "mcp_approval_response")
    )
}

pub(crate) fn prepare_mcp_request<'a>(
    request: &'a LlmRequest,
    target: WireFormat,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Cow<'a, LlmRequest>> {
    let responses_tools = mcp_tools(request).or_else(|| {
        (!is_anthropic_request(request))
            .then(|| {
                request
                    .preservation
                    .requests
                    .get(&WireFormat::OpenAiResponses.into())
                    .and_then(|body| body["tools"].as_array())
                    .filter(|tools| tools.iter().any(|tool| tool["type"] == "mcp"))
            })
            .flatten()
    });
    let anthropic_body = request
        .preservation
        .requests
        .get(&WireFormat::AnthropicMessages.into());
    let servers = is_anthropic_request(request)
        .then(|| request.extensions.fields.get("mcp_servers"))
        .flatten()
        .or_else(|| anthropic_body.and_then(|body| body.get("mcp_servers")))
        .and_then(Value::as_array)
        .filter(|servers| !servers.is_empty());
    if policy.target_capabilities.supports_tools == Some(false)
        && (responses_tools.is_some()
            || servers.is_some()
            || request
                .messages
                .iter()
                .flat_map(|message| &message.content)
                .any(|block| {
                    is_mcp_block(block, WireFormat::OpenAiResponses)
                        || is_mcp_block(block, WireFormat::AnthropicMessages)
                }))
    {
        push_lossy(
            diagnostics,
            policy,
            "target format/profile does not support MCP tools; MCP data dropped",
        )?;
        let mut prepared = request.clone();
        clear_mcp_preservation(&mut prepared);
        prepared.extensions.fields.remove(RESPONSES_MCP_TOOLS_KEY);
        prepared.extensions.fields.remove("mcp_servers");
        if let Some(tools) = prepared
            .extensions
            .fields
            .get_mut(ANTHROPIC_TOOLS_KEY)
            .and_then(Value::as_array_mut)
        {
            tools.retain(|tool| tool["type"] != "mcp_toolset");
        }
        prepared.messages.retain_mut(|message| {
            message.content.retain(|block| {
                !is_mcp_block(block, WireFormat::OpenAiResponses)
                    && !is_mcp_block(block, WireFormat::AnthropicMessages)
            });
            !message.content.is_empty()
        });
        prepared.tool_choice = None;
        return Ok(Cow::Owned(prepared));
    }
    // Inspect only the other provider's MCP data. Same-format settings and approvals
    // are replayed verbatim, and Anthropic-to-Chat loss uses the Anthropic tool validator.
    let source = if target == WireFormat::OpenAiResponses {
        WireFormat::AnthropicMessages
    } else {
        WireFormat::OpenAiResponses
    };
    let has_definitions = if source == WireFormat::AnthropicMessages {
        servers.is_some()
    } else {
        responses_tools.is_some()
    };
    if !has_definitions
        && !request
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .any(|block| is_mcp_block(block, source))
    {
        return Ok(Cow::Borrowed(request));
    }
    let mut prepared = request.clone();
    clear_mcp_preservation(&mut prepared);
    if target == WireFormat::OpenAiChat {
        push_lossy(
            diagnostics,
            policy,
            "Responses MCP tools and history are dropped for Chat Completions",
        )?;
        prepared.extensions.fields.remove(RESPONSES_MCP_TOOLS_KEY);
        prepared.messages.retain_mut(|message| {
            message
                .content
                .retain(|block| !is_mcp_block(block, WireFormat::OpenAiResponses));
            !message.content.is_empty()
        });
    } else if target == WireFormat::OpenAiResponses {
        let toolsets = request
            .extensions
            .fields
            .get(ANTHROPIC_TOOLS_KEY)
            .or_else(|| anthropic_body.and_then(|body| body.get("tools")))
            .and_then(Value::as_array);
        let mut tools = Vec::new();
        for server in servers.into_iter().flatten() {
            let toolset = toolsets.into_iter().flatten().find(|tool| {
                tool["type"] == "mcp_toolset" && tool["mcp_server_name"] == server["name"]
            });
            if let Some(tool) = anthropic_mcp_tool_to_responses(server, toolset) {
                tools.push(tool);
            } else {
                push_lossy(
                    diagnostics,
                    policy,
                    "Anthropic MCP server settings have no Responses equivalent; server dropped",
                )?;
            }
        }
        prepared
            .extensions
            .fields
            .insert(RESPONSES_MCP_TOOLS_KEY.to_string(), json!(tools));
        prepared.extensions.fields.remove("mcp_servers");
        if let Some(toolsets) = toolsets {
            prepared.extensions.fields.insert(
                ANTHROPIC_TOOLS_KEY.to_string(),
                json!(
                    toolsets
                        .iter()
                        .filter(|tool| tool["type"] != "mcp_toolset")
                        .collect::<Vec<_>>()
                ),
            );
        }
        prepared.messages =
            anthropic_mcp_history_to_responses(&request.messages, diagnostics, policy)?;
    } else {
        let mut servers = Vec::new();
        let mut toolsets = Vec::new();
        let tools = responses_tools
            .into_iter()
            .flatten()
            .filter(|tool| tool["type"] == "mcp")
            .cloned()
            .collect::<Vec<_>>();
        prepared
            .extensions
            .fields
            .insert(RESPONSES_MCP_TOOLS_KEY.to_string(), json!(tools));
        for tool in &tools {
            if let Some((server, toolset)) = responses_mcp_tool_to_anthropic(tool) {
                servers.push(server);
                toolsets.push(toolset);
            } else {
                // Messages has no MCP approval handshake. Dropping an approval requirement
                // would grant access the caller did not authorize, so drop the server instead.
                push_lossy(
                    diagnostics,
                    policy,
                    "Responses MCP server settings or approvals have no Anthropic equivalent; server dropped",
                )?;
            }
        }
        prepared
            .extensions
            .fields
            .insert("mcp_servers".to_string(), json!(servers));
        prepared
            .extensions
            .fields
            .insert(ANTHROPIC_TOOLS_KEY.to_string(), json!(toolsets));
        prepared.messages =
            responses_mcp_history_to_anthropic(&request.messages, diagnostics, policy)?;
    }
    if matches!(&prepared.tool_choice, Some(ToolChoice::Raw(choice)) if choice["type"] == "mcp")
        || matches!(&prepared.tool_choice, Some(ToolChoice::Tool { name })
            if !prepared.tools.iter().any(|tool| &tool.name == name)
                && !prepared.extensions.fields.get(ANTHROPIC_TOOLS_KEY).and_then(Value::as_array)
                    .is_some_and(|tools| tools.iter().any(|tool| tool["name"].as_str() == Some(name))))
    {
        push_lossy(
            diagnostics,
            policy,
            "MCP tool choice has no target equivalent; choice dropped",
        )?;
        prepared.tool_choice = None;
    }
    Ok(Cow::Owned(prepared))
}

fn clear_mcp_preservation(request: &mut LlmRequest) {
    // Clear all exact-replay bodies and embedded copies: either may restore a dropped
    // setting or forward credentials for a server absent from the target request.
    request.preservation = PreservationMetadata::default();
    if let Some(metadata) = request
        .extensions
        .fields
        .get_mut("metadata")
        .and_then(Value::as_object_mut)
    {
        metadata.remove(PRESERVATION_METADATA_KEY);
    }
}

fn mcp_config_enabled(config: &Value, default: bool) -> Option<bool> {
    if config.is_null() {
        return Some(default);
    }
    if !only_fields(config, &["enabled", "defer_loading"])
        || (!config["defer_loading"].is_null() && config["defer_loading"] != false)
    {
        return None;
    }
    if config["enabled"].is_null() {
        Some(default)
    } else {
        config["enabled"].as_bool()
    }
}

pub(super) fn anthropic_mcp_tool_to_responses(
    server: &Value,
    toolset: Option<&Value>,
) -> Option<Value> {
    if !only_fields(
        server,
        &[
            "type",
            "name",
            "url",
            "authorization_token",
            "tool_configuration",
        ],
    ) || server["type"] != "url"
    {
        return None;
    }
    let mut tool = json!({"type": "mcp", "server_label": server["name"].as_str()?,
        "server_url": server["url"].as_str()?, "require_approval": "never"});
    if let Some(token) = server.get("authorization_token") {
        tool["authorization"] = token.clone();
    }
    if let Some(toolset) = toolset {
        if !only_fields(
            toolset,
            &["type", "mcp_server_name", "default_config", "configs"],
        ) {
            return None;
        }
        let enabled = mcp_config_enabled(&toolset["default_config"], true)?;
        let mut allowed = Vec::new();
        if !toolset["configs"].is_null() {
            for (name, config) in toolset["configs"].as_object()? {
                let tool_enabled = mcp_config_enabled(config, enabled)?;
                // A denylist cannot be turned into an allowlist without a complete tool listing.
                if enabled && !tool_enabled {
                    return None;
                }
                if tool_enabled {
                    allowed.push(json!(name));
                }
            }
        }
        if !enabled {
            tool["allowed_tools"] = json!(allowed);
        }
        if !server["tool_configuration"].is_null() {
            return None;
        }
    } else if !server["tool_configuration"].is_null() {
        let config = &server["tool_configuration"];
        if !only_fields(config, &["enabled", "allowed_tools"]) {
            return None;
        }
        if config["enabled"] == false {
            tool["allowed_tools"] = json!([]);
        } else if let Some(allowed) = config.get("allowed_tools") {
            tool["allowed_tools"] = allowed.clone();
        }
    }
    Some(tool)
}

pub(super) fn anthropic_mcp_history_to_responses(
    messages: &[Message],
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<Message>> {
    let mut translated = Vec::new();
    let mut results = HashMap::new();
    let mut calls = HashSet::new();
    for block in messages.iter().flat_map(|message| &message.content) {
        if let ContentBlock::Unknown { provider, raw } = block
            && provider.as_str() == WireFormat::AnthropicMessages.as_str()
        {
            match raw["type"].as_str() {
                Some("mcp_tool_result") => {
                    results.insert(raw["tool_use_id"].clone(), raw);
                }
                Some("mcp_tool_use") => {
                    calls.insert(raw["id"].clone());
                }
                _ => {}
            }
        }
    }
    for message in messages {
        let mut content = Vec::new();
        for block in &message.content {
            if !is_mcp_block(block, WireFormat::AnthropicMessages) {
                content.push(block.clone());
                continue;
            }
            let ContentBlock::Unknown { raw, .. } = block else {
                unreachable!()
            };
            if raw["type"] == "mcp_tool_result" {
                if !calls.contains(&raw["tool_use_id"]) {
                    push_lossy(
                        diagnostics,
                        policy,
                        "Anthropic MCP result has no matching call; result dropped",
                    )?;
                }
                continue;
            }
            let result = results.get(&raw["id"]);
            let output = result.and_then(|result| {
                result["content"].as_str().map(str::to_owned).or_else(|| {
                    result["content"]
                        .as_array()
                        .filter(|blocks| {
                            blocks
                                .iter()
                                .all(|block| block["type"] == "text" && block["text"].is_string())
                        })
                        .map(|blocks| {
                            blocks
                                .iter()
                                .map(|block| block["text"].as_str().unwrap())
                                .collect::<Vec<_>>()
                                .join("\n")
                        })
                })
            });
            if result.is_none_or(|result| {
                !only_fields(result, &["type", "tool_use_id", "is_error", "content"])
            }) || !only_fields(raw, &["type", "id", "name", "server_name", "input"])
                || output.is_none()
            {
                push_lossy(
                    diagnostics,
                    policy,
                    "Anthropic MCP incomplete call or result content has no Responses equivalent; call dropped",
                )?;
                continue;
            }
            let result = result.unwrap();
            let is_error = result["is_error"] == true;
            if !content.is_empty() {
                translated.push(Message {
                    role: message.role,
                    content: std::mem::take(&mut content),
                });
            }
            translated.push(Message {role: Role::Assistant, content: vec![ContentBlock::Unknown {
                provider: WireFormat::OpenAiResponses.into(), raw: json!({"type": "mcp_call", "id": raw["id"],
                    "server_label": raw["server_name"], "name": raw["name"], "arguments": raw["input"].to_string(),
                    "output": if is_error {Value::Null} else {json!(output.unwrap())}, "approval_request_id": null,
                    "error": if is_error {json!({"type": "mcp_tool_execution_error", "content": result["content"]})} else {Value::Null},
                    "status": if is_error {"failed"} else {"completed"}})
            }]});
        }
        if !content.is_empty() {
            translated.push(Message {
                role: message.role,
                content,
            });
        }
    }
    Ok(translated)
}

pub(crate) fn is_native_output(kind: &str) -> bool {
    matches!(
        kind,
        "apply_patch_call" | "shell_call" | "computer_call" | "image_generation_call"
    )
}

pub(crate) fn validate_response_output(
    response: &crate::AggLlmResponse,
    target: crate::WireFormat,
) -> crate::Result<()> {
    for block in response.outputs.iter().flat_map(|output| &output.content) {
        if let crate::ContentBlock::Unknown { provider, raw } = block
            && provider.as_str() == crate::WireFormat::OpenAiResponses.as_str()
            && raw["type"].as_str().is_some_and(is_native_output)
        {
            return Err(crate::TranslationError::UnsupportedTranslation {
                from: provider.clone(),
                to: target.into(),
            });
        }
    }
    Ok(())
}

// Native outputs have no neutral streaming representation; inspect them before discarding raw events.
pub(crate) fn validate_stream_output(
    source: &crate::FormatId,
    target: &crate::FormatId,
    event: &serde_json::Value,
) -> crate::Result<()> {
    if source.as_str() == crate::WireFormat::OpenAiResponses.as_str() && source != target {
        let mut items = event
            .get("item")
            .into_iter()
            .chain(event["response"]["output"].as_array().into_iter().flatten());
        if items.any(|item| item["type"].as_str().is_some_and(is_native_output)) {
            return Err(crate::TranslationError::UnsupportedTranslation {
                from: source.clone(),
                to: target.clone(),
            });
        }
    }
    Ok(())
}

// This is a transport envelope, not encryption. Keep it distinct from native OpenAI
// ciphertext, and retain the original text because the signature covers that text.
const ANTHROPIC_THINKING_PREFIX: &str = "switchyard:anthropic-thinking:v1:";

fn encode_anthropic_thinking(text: &str, signature: &str) -> String {
    format!(
        "{ANTHROPIC_THINKING_PREFIX}{}",
        serde_json::json!([text, signature])
    )
}

fn decode_anthropic_thinking(payload: &str) -> Option<(String, String)> {
    serde_json::from_str(payload.strip_prefix(ANTHROPIC_THINKING_PREFIX)?).ok()
}
