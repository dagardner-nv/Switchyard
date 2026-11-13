// BEGIN TEMPORARY TEST: Codex end-to-end tool diagnostics. Do not commit.
use serde_json::Value;

pub fn tool_names(tools: &Value) -> Vec<String> {
    fn collect(tools: &Value, namespace: &str, output: &mut Vec<String>) {
        if let Some(tools) = tools.as_array() {
            for tool in tools {
                let name = tool["name"]
                    .as_str()
                    .or_else(|| tool["function"]["name"].as_str())
                    .or_else(|| tool["type"].as_str())
                    .unwrap_or("<unnamed>");
                let qualified = format!("{namespace}{name}");
                if tool["type"] == "namespace" {
                    collect(&tool["tools"], &format!("{qualified}/"), output);
                } else {
                    output.push(format!("\x1b[35m{qualified}\x1b[0m"));
                }
            }
        }
    }
    let mut names = Vec::new();
    collect(tools, "", &mut names);
    names
}

pub fn format_response(body: &Value, label: &str) -> Option<String> {
    fn collect(value: &Value, names: &mut Vec<String>) -> bool {
        match value {
            Value::Object(object) => {
                let kind = value["type"].as_str().unwrap_or("");
                let mut has_call =
                    matches!(kind, "function_call" | "custom_tool_call" | "tool_use")
                        || kind.starts_with("response.function_call_arguments.")
                        || value["tool_calls"]
                            .as_array()
                            .is_some_and(|calls| !calls.is_empty());
                if let Some(name) = value["function"]["name"]
                    .as_str()
                    .or_else(|| has_call.then(|| value["name"].as_str()).flatten())
                {
                    names.push(name.to_owned());
                    has_call = true;
                }
                for child in object.values() {
                    has_call |= collect(child, names);
                }
                has_call
            }
            Value::Array(items) => {
                let mut has_call = false;
                for item in items {
                    has_call |= collect(item, names);
                }
                has_call
            }
            _ => false,
        }
    }
    let mut names = Vec::new();
    if !collect(body, &mut names) {
        return None;
    }
    names.sort();
    names.dedup();
    let mut pretty = serde_json::to_string_pretty(body).ok()?;
    for name in names {
        let quoted = serde_json::to_string(&name).ok()?;
        pretty = pretty.replace(
            &format!("\"name\": {quoted}"),
            &format!("\"name\": \x1b[32m{quoted}\x1b[0m"),
        );
    }
    Some(format!("{label}:\n{pretty}"))
}
// END TEMPORARY TEST.
