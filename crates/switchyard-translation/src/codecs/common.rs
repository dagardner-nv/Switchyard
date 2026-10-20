// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-agnostic helpers shared by wire-format codecs.

use std::borrow::Cow;

use serde_json::{Map, Value};

use crate::error::{Result, TranslationError};
use crate::llm::{ContentBlock, LlmRequest, Tool, ToolChoice};
use crate::{PreservationMetadata, TranslationDiagnostic, TranslationPolicy, WireFormat};

// Internal provenance survives mutations that invalidate exact request replay.
pub(crate) const ANTHROPIC_REQUEST_KEY: &str = "switchyard_anthropic_request";

/// Returns true when the request was decoded from an Anthropic Messages body.
pub(crate) fn is_anthropic_request(request: &LlmRequest) -> bool {
    request.extensions.fields.get(ANTHROPIC_REQUEST_KEY) == Some(&Value::Bool(true))
}

/// Converts an OpenAI allowed-tools policy to a restricted function list and mode.
pub(crate) fn allowed_function_tools(
    request: &LlmRequest,
) -> Result<Option<(Vec<Tool>, ToolChoice)>> {
    let Some(ToolChoice::Raw(choice)) = &request.tool_choice else {
        return Ok(None);
    };
    if choice.get("type").and_then(Value::as_str) != Some("allowed_tools") {
        return Ok(None);
    }
    let error = || {
        TranslationError::LossyConversion(
            "allowed_tools translation requires auto or required mode \
             and a nonempty subset of known function tools"
                .to_string(),
        )
    };
    let policy = choice.get("allowed_tools").unwrap_or(choice);
    let mode = match policy.get("mode").and_then(Value::as_str) {
        Some("auto") => ToolChoice::Auto,
        Some("required") => ToolChoice::Required,
        _ => return Err(error()),
    };
    let allowed = policy
        .get("tools")
        .and_then(Value::as_array)
        .ok_or_else(error)?;
    if allowed.is_empty() {
        return Err(error());
    }
    let mut names = Vec::with_capacity(allowed.len());
    for tool in allowed {
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            return Err(error());
        }
        let function = tool.get("function").unwrap_or(tool);
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(error)?;
        let name = match function.get("namespace").and_then(Value::as_str) {
            Some(namespace) if !namespace.is_empty() => {
                crate::codex_namespaces::qualified_tool_name(namespace, name)
            }
            _ => name.to_string(),
        };
        if !request
            .tools
            .iter()
            .filter_map(Tool::function)
            .any(|tool| tool.name == name)
        {
            return Err(error());
        }
        names.push(name);
    }
    // Filter the catalog, rather than rebuilding it from selectors, to retain definitions
    // and avoid duplicating tools when the allowed list repeats a name.
    let tools = request
        .tools
        .iter()
        .filter(|tool| {
            tool.function()
                .is_some_and(|tool| names.contains(&tool.name))
        })
        .cloned()
        .collect();
    Ok(Some((tools, mode)))
}

/// Returns whether a role name is recognized by a supported provider API.
pub(crate) fn is_known_role_name(name: &str) -> bool {
    matches!(
        name,
        "system" | "developer" | "user" | "assistant" | "tool" | "function"
    )
}

/// Extracts text-like blocks and joins them for text-only provider fields.
pub(crate) fn text_from_blocks(content: &[ContentBlock], separator: &str) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            ContentBlock::Refusal { text } => Some(text.as_str()),
            ContentBlock::Unknown { raw, .. } => raw.as_str(),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(separator)
}

/// Extracts private reasoning blocks without mixing them into visible text.
pub(crate) fn reasoning_text_from_blocks(content: &[ContentBlock], separator: &str) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Reasoning { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(separator)
}

/// Extracts displayable text from structured reasoning details.
pub(crate) fn reasoning_text_from_details(details: &[Value]) -> Option<String> {
    let parts = details
        .iter()
        .filter_map(Value::as_object)
        .filter_map(|detail| {
            detail
                .get("text")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .or_else(|| {
                    detail
                        .get("summary")
                        .and_then(Value::as_str)
                        .filter(|summary| !summary.is_empty())
                })
        })
        .collect::<Vec<_>>();
    (!parts.is_empty()).then(|| parts.join("\n"))
}

/// Collects reasoning text from a Responses reasoning item's `content` or `summary`
/// array, or from a bare string, into `out`. Empty strings are skipped.
pub(crate) fn collect_responses_reasoning_text(value: Option<&Value>, out: &mut Vec<String>) {
    match value {
        Some(Value::String(text)) if !text.is_empty() => out.push(text.clone()),
        Some(Value::Array(items)) => {
            for item in items {
                match item {
                    Value::String(text) if !text.is_empty() => out.push(text.clone()),
                    Value::Object(object) => {
                        if matches!(
                            object.get("type").and_then(Value::as_str),
                            Some("reasoning_text" | "summary_text" | "text")
                        ) && let Some(text) = object.get("text").and_then(Value::as_str)
                            && !text.is_empty()
                        {
                            out.push(text.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

/// Returns the opaque payload of the first encrypted reasoning detail, if any.
///
/// Two detail shapes are accepted: the documented `{"type": "reasoning.encrypted", "data"}`
/// object, and a verbatim Responses `reasoning` item carrying `encrypted_content` (the shape
/// the buffered request decoder stores when it keeps the provider item whole).
pub(crate) fn encrypted_reasoning_data(details: &[Value]) -> Option<String> {
    details
        .iter()
        .filter_map(Value::as_object)
        .find_map(|detail| match detail.get("type").and_then(Value::as_str) {
            Some("reasoning.encrypted") => detail.get("data").and_then(Value::as_str),
            Some("reasoning") => detail.get("encrypted_content").and_then(Value::as_str),
            _ => None,
        })
        .filter(|data| !data.is_empty())
        .map(ToOwned::to_owned)
}

/// Returns the provider item id recorded on the first `reasoning.encrypted` detail, if any.
/// Encrypted reasoning is bound to the item id it was issued under, so a replay must reuse it.
pub(crate) fn encrypted_reasoning_item_id(details: &[Value]) -> Option<String> {
    details
        .iter()
        .filter_map(Value::as_object)
        .find(|detail| {
            matches!(
                detail.get("type").and_then(Value::as_str),
                Some("reasoning.encrypted" | "reasoning")
            )
        })
        .and_then(|detail| detail.get("id").and_then(Value::as_str))
        .filter(|id| !id.is_empty())
        .map(ToOwned::to_owned)
}

/// Returns the first non-empty string stored under the requested keys.
pub(crate) fn first_nonempty_string<'a>(
    object: &'a Map<String, Value>,
    keys: &[&str],
) -> Option<&'a str> {
    keys.iter().find_map(|key| {
        object
            .get(*key)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
    })
}

/// Copies unknown provider fields into the IR extension map.
pub(crate) fn provider_extensions(
    object: &Map<String, Value>,
    known: &[&str],
) -> Map<String, Value> {
    let mut extensions = Map::new();
    for (key, value) in object {
        if key != ANTHROPIC_REQUEST_KEY && !known.contains(&key.as_str()) {
            extensions.insert(key.clone(), value.clone());
        }
    }
    extensions
}

// Apply policy before saved bodies or embedded metadata can restore dropped tools.
pub(crate) fn prepare_request_tools<'a>(
    request: &'a LlmRequest,
    target: WireFormat,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Cow<'a, LlmRequest>> {
    let has_unsupported = request
        .tools
        .iter()
        .any(|tool| !matches!(tool, Tool::Function(_)))
        || request_content(request).any(is_mcp_content)
        || matches!(&request.tool_choice, Some(ToolChoice::Raw(raw)) if raw["type"] == "mcp");
    let has_preserved_mcp = request
        .preservation
        .requests
        .values()
        .any(preserved_request_has_mcp)
        || request.extensions.fields.contains_key("mcp_servers")
        || has_mcp_tools(
            request
                .extensions
                .fields
                .get(crate::codex_custom_tools::ADDITIONAL_TOOLS_KEY)
                .unwrap_or(&Value::Null),
        )
        || request
            .extensions
            .fields
            .get("metadata")
            .and_then(|metadata| metadata.get(crate::PRESERVATION_METADATA_KEY))
            .and_then(|envelope| envelope.get("requests"))
            .and_then(Value::as_object)
            .is_some_and(|requests| requests.values().any(preserved_request_has_mcp));
    if !has_unsupported && !has_preserved_mcp {
        return Ok(Cow::Borrowed(request));
    }
    // Native replay can retain opaque definitions that these codecs cannot rebuild.
    if target != WireFormat::OpenAiChat
        && policy.preservation != crate::PreservationPolicy::Disabled
        && can_replay_mcp_request(request, target)
    {
        return Ok(Cow::Borrowed(request));
    }
    crate::util::push_lossy(
        diagnostics,
        policy,
        format!("unmapped MCP or provider tools and history are dropped for {target}"),
    )?;
    let mut request = request.clone();
    request
        .tools
        .retain(|tool| matches!(tool, Tool::Function(_)));
    for message in &mut request.messages {
        drop_mcp_content(&mut message.content);
    }
    request
        .messages
        .retain(|message| !message.content.is_empty());
    for instruction in &mut request.instructions {
        drop_mcp_content(&mut instruction.content);
    }
    request
        .instructions
        .retain(|instruction| !instruction.content.is_empty());
    if request.tools.is_empty()
        || matches!(&request.tool_choice, Some(ToolChoice::Raw(raw)) if raw["type"] == "mcp")
        || matches!(&request.tool_choice, Some(ToolChoice::Tool { name })
            if !request.tools.iter().filter_map(Tool::function).any(|tool| &tool.name == name))
    {
        request.tool_choice = None;
    }
    // All cached formats must be cleared; another snapshot can hold older credentials.
    request.preservation = PreservationMetadata::default();
    request.extensions.fields.remove("mcp_servers");
    request
        .extensions
        .fields
        .remove(crate::codex_custom_tools::ADDITIONAL_TOOLS_KEY);
    if let Some(metadata) = request
        .extensions
        .fields
        .get_mut("metadata")
        .and_then(Value::as_object_mut)
    {
        metadata.remove(crate::PRESERVATION_METADATA_KEY);
    }
    Ok(Cow::Owned(request))
}

fn is_mcp_content(block: &ContentBlock) -> bool {
    match block {
        ContentBlock::Mcp(_) => true,
        ContentBlock::Unknown { raw, .. } => raw["type"]
            .as_str()
            .is_some_and(|kind| kind.starts_with("mcp_")),
        ContentBlock::ToolResult(result) => result.content.iter().any(is_mcp_content),
        _ => false,
    }
}

fn drop_mcp_content(content: &mut Vec<ContentBlock>) {
    content.retain_mut(|block| {
        if let ContentBlock::ToolResult(result) = block {
            drop_mcp_content(&mut result.content);
        }
        !is_mcp_content(block)
    });
}

fn request_content(request: &LlmRequest) -> impl Iterator<Item = &ContentBlock> {
    request
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .chain(
            request
                .instructions
                .iter()
                .flat_map(|instruction| &instruction.content),
        )
}

fn has_mcp_tools(tools: &Value) -> bool {
    tools.as_array().is_some_and(|tools| {
        tools.iter().any(|tool| {
            matches!(tool["type"].as_str(), Some("mcp" | "mcp_toolset"))
                || (tool["type"] == "namespace" && has_mcp_tools(&tool["tools"]))
        })
    })
}

fn has_mcp_history(items: &Value) -> bool {
    items.as_array().is_some_and(|items| {
        items.iter().any(|item| {
            item["type"]
                .as_str()
                .is_some_and(|kind| kind.starts_with("mcp_"))
        })
    })
}

fn preserved_request_has_mcp(body: &Value) -> bool {
    body.get("mcp_servers").is_some()
        || has_mcp_tools(&body["tools"])
        || has_mcp_history(&body["input"])
        || body["input"].as_array().is_some_and(|items| {
            items
                .iter()
                .any(|item| item["type"] == "additional_tools" && has_mcp_tools(&item["tools"]))
        })
        || body["messages"].as_array().is_some_and(|messages| {
            messages
                .iter()
                .any(|message| has_mcp_history(&message["content"]))
        })
}

// Multiple snapshots can contain stale credentials or history. Comparing only the
// target snapshot cannot make the other snapshots safe to embed, so clear them all.
fn can_replay_mcp_request(request: &LlmRequest, target: WireFormat) -> bool {
    request.preservation.requests.len() == 1
        && request
            .preservation
            .requests
            .get(&target.into())
            .is_some_and(|body| {
                // An embedded envelope can carry snapshots beyond the one checked here.
                body["metadata"][crate::PRESERVATION_METADATA_KEY].is_null()
                    && preservation_codec(target)
                        .decode_request(body, &preservation_check_policy())
                        .is_ok_and(|original| {
                            original.request.tools == request.tools
                                && original.request.messages == request.messages
                                && instructions_match(&original.request, request)
                                && original.request.tool_choice == request.tool_choice
                                && original.request.extensions.fields == request.extensions.fields
                        })
            })
}

fn instructions_match(original: &LlmRequest, request: &LlmRequest) -> bool {
    if original.instructions == request.instructions {
        return true;
    }
    // Target prompts create separate IR blocks, but the wire body may combine
    // them into one system instruction. Compare the text those encoders emit.
    let system_text = |request: &LlmRequest| {
        let mut parts = Vec::new();
        for instruction in &request.instructions {
            if instruction.role != crate::llm::Role::System {
                return None;
            }
            for block in &instruction.content {
                let ContentBlock::Text { text } = block else {
                    return None;
                };
                parts.push(text.as_str());
            }
        }
        Some(parts.join("\n\n"))
    };
    system_text(original).is_some_and(|text| Some(text) == system_text(request))
}

fn preservation_codec(format: WireFormat) -> &'static dyn super::FormatCodec {
    match format {
        WireFormat::AnthropicMessages => &super::anthropic::AnthropicMessagesCodec,
        WireFormat::OpenAiResponses => &super::responses::OpenAiResponsesCodec,
        WireFormat::OpenAiChat => &super::openai_chat::OpenAiChatCodec,
    }
}

fn preservation_check_policy() -> TranslationPolicy {
    TranslationPolicy {
        preservation: crate::PreservationPolicy::Disabled,
        ..TranslationPolicy::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn encrypted_reasoning_helpers_accept_both_detail_shapes() {
        let documented = vec![json!({"type": "reasoning.encrypted", "data": "blob", "id": "rs_1"})];
        assert_eq!(
            encrypted_reasoning_data(&documented).as_deref(),
            Some("blob")
        );
        assert_eq!(
            encrypted_reasoning_item_id(&documented).as_deref(),
            Some("rs_1")
        );
        let verbatim_item = vec![json!({
            "type": "reasoning", "id": "rs_2", "summary": [], "encrypted_content": "blob2"
        })];
        assert_eq!(
            encrypted_reasoning_data(&verbatim_item).as_deref(),
            Some("blob2")
        );
        assert_eq!(
            encrypted_reasoning_item_id(&verbatim_item).as_deref(),
            Some("rs_2")
        );
        assert_eq!(encrypted_reasoning_data(&[json!({"type": "other"})]), None);
    }
}
