// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-agnostic helpers shared by wire-format codecs.

use std::borrow::Cow;

use serde_json::{Map, Value};

use crate::error::{Result, TranslationError};
use crate::llm::{ContentBlock, LlmRequest, McpContent, Tool, ToolChoice};
use crate::{
    AggLlmResponse, PreservationMetadata, TranslationDiagnostic, TranslationPolicy, WireFormat,
};

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
                .is_some_and(|definition| names.contains(&definition.name))
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
        if key != ANTHROPIC_REQUEST_KEY
            && key != super::anthropic::MCP_LAYOUTS_KEY
            && !known.contains(&key.as_str())
        {
            extensions.insert(key.clone(), value.clone());
        }
    }
    extensions
}

// Remove unsupported tool data before exact replay or preservation embedding can restore it.
pub(crate) fn prepare_request_tools<'a>(
    request: &'a LlmRequest,
    target: WireFormat,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Cow<'a, LlmRequest>> {
    let unsupported_tool = |tool: &Tool| match tool {
        Tool::Function(_) => false,
        Tool::Mcp(mcp) => {
            target != WireFormat::AnthropicMessages
                || !super::anthropic::can_encode_mcp(mcp, request)
        }
        Tool::Provider { provider, .. } => provider.as_str() != target.as_str(),
    };
    let preserved_tools = target != WireFormat::AnthropicMessages
        && (request
            .extensions
            .fields
            .contains_key(super::anthropic::MCP_LAYOUTS_KEY)
            || request
                .preservation
                .requests
                .get(&WireFormat::AnthropicMessages.into())
                .is_some_and(|body| {
                    body.get("mcp_servers")
                        .and_then(Value::as_array)
                        .is_some_and(|v| !v.is_empty())
                        || body
                            .get("tools")
                            .and_then(Value::as_array)
                            .is_some_and(|tools| {
                                tools
                                    .iter()
                                    .any(|tool| !super::anthropic::is_function_tool(tool))
                            })
                }));
    if !request.tools.iter().any(unsupported_tool)
        && !preserved_tools
        && !request
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .chain(
                request
                    .instructions
                    .iter()
                    .flat_map(|instruction| &instruction.content),
            )
            .any(|block| is_unsupported_tool_content(block, target))
    {
        return Ok(Cow::Borrowed(request));
    }
    crate::util::push_lossy(
        diagnostics,
        policy,
        format!("unmapped MCP or provider tools and history are dropped for {target}"),
    )?;
    let mut request = request.clone();
    request.tools.retain(|tool| !unsupported_tool(tool));
    if target != WireFormat::AnthropicMessages {
        request
            .extensions
            .fields
            .remove(super::anthropic::MCP_LAYOUTS_KEY);
    }
    request.messages.retain_mut(|message| {
        drop_unsupported_tool_content(&mut message.content, target);
        !message.content.is_empty()
    });
    request.instructions.retain_mut(|instruction| {
        drop_unsupported_tool_content(&mut instruction.content, target);
        !instruction.content.is_empty()
    });
    if request.tools.is_empty()
        || matches!(&request.tool_choice, Some(ToolChoice::Tool { name })
        if !request.tools.iter().any(|tool| match tool {
            Tool::Function(definition) => &definition.name == name,
            Tool::Provider { raw, .. } => raw.get("name").and_then(Value::as_str) == Some(name),
            Tool::Mcp(_) => false,
        }))
    {
        request.tool_choice = None;
    }
    // Clear every format's cached body and embedded envelope: either could restore credentials.
    request.preservation = PreservationMetadata::default();
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

fn is_unsupported_tool_content(block: &ContentBlock, target: WireFormat) -> bool {
    match block {
        ContentBlock::Mcp(_) => target != WireFormat::AnthropicMessages,
        ContentBlock::Unknown { provider, raw } => {
            provider.as_str() == WireFormat::AnthropicMessages.as_str()
                && target != WireFormat::AnthropicMessages
                && super::anthropic::is_provider_tool_block(raw)
        }
        ContentBlock::ToolResult(result) => result
            .content
            .iter()
            .any(|block| is_unsupported_tool_content(block, target)),
        _ => false,
    }
}
fn drop_unsupported_tool_content(content: &mut Vec<ContentBlock>, target: WireFormat) {
    content.retain_mut(|block| {
        match block {
            ContentBlock::ToolResult(result) | ContentBlock::Mcp(McpContent::Result(result)) => {
                drop_unsupported_tool_content(&mut result.content, target)
            }
            _ => {}
        }
        !is_unsupported_tool_content(block, target)
    });
}

pub(crate) fn prepare_response_tools<'a>(
    response: &'a AggLlmResponse,
    target: WireFormat,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Cow<'a, AggLlmResponse>> {
    if !response
        .outputs
        .iter()
        .flat_map(|output| &output.content)
        .any(|block| matches!(block, ContentBlock::Mcp(_)))
    {
        return Ok(Cow::Borrowed(response));
    }
    crate::util::push_lossy(
        diagnostics,
        policy,
        "unmapped MCP or provider tool output is dropped",
    )?;
    let mut response = response.clone();
    for output in &mut response.outputs {
        drop_unsupported_tool_content(&mut output.content, target);
    }
    response.preservation = PreservationMetadata::default();
    if let Some(metadata) = response
        .extensions
        .fields
        .get_mut("metadata")
        .and_then(Value::as_object_mut)
    {
        metadata.remove(crate::PRESERVATION_METADATA_KEY);
    }
    Ok(Cow::Owned(response))
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
