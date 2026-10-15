//! Every indexed tool call in an ordinary Chat Completions delta must retain its arguments.

use futures::StreamExt;
use genai::adapter::AdapterKind;
use genai::chat::{ChatOptions, ChatRequest, ChatStreamEvent};
use genai::resolver::{AuthData, Endpoint};
use genai::{Client, ModelIden, ServiceTarget};
use serde_json::{Value, json};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

fn chunk(tool_calls: Vec<Value>, finish_reason: Option<&str>) -> Value {
	json!({
		"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 0, "model": "gpt-4o",
		"choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": finish_reason}]
	})
}

async fn captured_calls(combined_arguments: bool) -> Vec<(String, String, Value)> {
	let mut events = Vec::new();
	for index in 0..2 {
		events.push(chunk(
			vec![json!({
				"index": index, "id": format!("call_{index}"), "type": "function",
				"function": {"name": "get_weather", "arguments": ""}
			})],
			None,
		));
	}
	let deltas: Vec<_> = ["Paris", "Tokyo"]
		.into_iter()
		.enumerate()
		.map(|(index, city)| json!({"index": index, "function": {"arguments": json!({"city": city}).to_string()}}))
		.collect();
	if combined_arguments {
		events.push(chunk(deltas, None));
	} else {
		for delta in deltas {
			events.push(chunk(vec![delta], None));
		}
	}
	events.push(chunk(vec![], Some("tool_calls")));
	let mut body = events.iter().map(|event| format!("data: {event}\n\n")).collect::<String>();
	body.push_str("data: [DONE]\n\n");

	let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
	let target = ServiceTarget {
		endpoint: Endpoint::from_owned(format!("http://{}/v1/", listener.local_addr().unwrap())),
		auth: AuthData::from_single("test"),
		model: ModelIden::new(AdapterKind::OpenAI, "gpt-4o"),
	};
	let server = tokio::spawn(async move {
		let (mut socket, _) = listener.accept().await.unwrap();
		let mut request = Vec::new();
		let mut buf = [0; 4096];
		loop {
			let n = socket.read(&mut buf).await.unwrap();
			assert_ne!(n, 0);
			request.extend_from_slice(&buf[..n]);
			if let Some(end) = request.windows(4).position(|part| part == b"\r\n\r\n") {
				let headers = String::from_utf8_lossy(&request[..end]);
				let length: usize = headers
					.lines()
					.find_map(|line| {
						line.to_lowercase()
							.strip_prefix("content-length:")
							.map(|s| s.trim().parse().unwrap())
					})
					.unwrap();
				if request.len() >= end + 4 + length {
					break;
				}
			}
		}
		socket
			.write_all(
				format!(
					"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
					body.len()
				)
				.as_bytes(),
			)
			.await
			.unwrap();
	});
	let client = Client::builder()
		.with_reqwest(reqwest::Client::builder().no_proxy().build().unwrap())
		.build()
		.unwrap();
	let options = ChatOptions::default().with_capture_tool_calls(true);
	let calls = tokio::time::timeout(Duration::from_secs(5), async {
		let response = client
			.exec_chat_stream(target, ChatRequest::from_user("weather"), Some(&options))
			.await
			.unwrap();
		let mut stream = response.stream;
		while let Some(event) = stream.next().await {
			if let ChatStreamEvent::End(end) = event.unwrap() {
				return end
					.captured_tool_calls()
					.unwrap()
					.iter()
					.map(|call| (call.call_id.clone(), call.fn_name.clone(), call.fn_arguments.clone()))
					.collect();
			}
		}
		panic!("missing stream end");
	})
	.await
	.expect("stream must finish");
	server.await.unwrap();
	calls
}

fn expected_calls() -> Vec<(String, String, Value)> {
	vec![
		("call_0".into(), "get_weather".into(), json!({"city": "Paris"})),
		("call_1".into(), "get_weather".into(), json!({"city": "Tokyo"})),
	]
}

#[tokio::test]
async fn all_tool_arguments_in_one_ordinary_frame_are_captured() {
	assert_eq!(captured_calls(true).await, expected_calls());
}

#[tokio::test]
async fn tool_arguments_in_separate_ordinary_frames_are_captured() {
	assert_eq!(captured_calls(false).await, expected_calls());
}
