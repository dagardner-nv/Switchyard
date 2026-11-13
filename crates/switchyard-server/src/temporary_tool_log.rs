// BEGIN TEMPORARY TEST: Codex end-to-end tool diagnostics. Do not commit.
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;
use switchyard_llm_client::temporary_tool_log::{format_response, tool_names};

use crate::redaction::Redactor;

static REQUEST_NUMBER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn request(body: &Value) {
    let tools = tool_names(&body["tools"]);
    let number = REQUEST_NUMBER.fetch_add(1, Ordering::Relaxed) + 1;
    let separator = if number > 1 {
        "\n-----------------\n"
    } else {
        ""
    };
    eprintln!("{separator}Request {number} tools: {}", tools.join(", "));
}

pub(crate) fn response(body: &Value, redactor: &Redactor) {
    if let Some(log) = format_response(body, "Response encoded tool_call (Responses)") {
        eprintln!("--->\n{}", redactor.text(log));
    }
}
// END TEMPORARY TEST.
