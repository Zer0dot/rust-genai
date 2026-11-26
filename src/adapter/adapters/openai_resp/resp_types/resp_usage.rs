use crate::chat::{CompletionTokensDetails, PromptTokensDetails, Usage};
use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;

/// Normalized token usage across providers (OpenAI-compatible).
///
/// - Cache counters preserve reported zeroes; absent counters remain `None`. Other detail counters treat 0 as `None`.
///
/// - `prompt_tokens` and `output_tokens` are the total input/output tokens. The corresponding `*_details` carry provider-specific breakdowns.
///
/// - Gemini: `candidatesTokenCount` excludes "thoughts" (reasoning) tokens. We normalize:
///   `output_tokens = candidatesTokenCount + thoughts_token_count`, and
///   `output_tokens_details.reasoning_tokens = thoughts_token_count`.
#[skip_serializing_none]
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct RespUsage {
	/// Total input tokens (formerly `input_tokens`).
	pub input_tokens: Option<i32>,
	pub input_tokens_details: Option<InputTokensDetails>,

	/// Total output (completion) tokens.
	pub output_tokens: Option<i32>,
	pub output_tokens_details: Option<OutputTokensDetails>,

	/// Total tokens as reported by the API, or computed as prompt + completion
	/// (including cache read/creation tokens when applicable).
	pub total_tokens: Option<i32>,
}

impl RespUsage {
	/// Remove detail objects that contain only `None` fields.
	pub fn compact_details(&mut self) {
		if self.input_tokens_details.as_ref().is_some_and(|d| d.is_empty()) {
			self.input_tokens_details = None;
		}

		if self.output_tokens_details.as_ref().is_some_and(|d| d.is_empty()) {
			self.output_tokens_details = None;
		}
	}
}

#[skip_serializing_none]
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct InputTokensDetails {
	/// OpenAI: tokens written to the prompt cache.
	#[serde(default)]
	pub cache_write_tokens: Option<i32>,
	/// OpenAI: tokens read from the prompt cache.
	#[serde(default)]
	pub cached_tokens: Option<i32>,
	#[serde(default, deserialize_with = "crate::support::zero_as_none")]
	pub audio_tokens: Option<i32>,
}

impl InputTokensDetails {
	/// True if all fields are `None`.
	pub fn is_empty(&self) -> bool {
		self.cache_write_tokens.is_none() && self.cached_tokens.is_none() && self.audio_tokens.is_none()
	}
}

#[skip_serializing_none]
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct OutputTokensDetails {
	#[serde(default, deserialize_with = "crate::support::zero_as_none")]
	pub accepted_prediction_tokens: Option<i32>,
	#[serde(default, deserialize_with = "crate::support::zero_as_none")]
	pub rejected_prediction_tokens: Option<i32>,
	#[serde(default, deserialize_with = "crate::support::zero_as_none")]
	pub reasoning_tokens: Option<i32>,
	#[serde(default, deserialize_with = "crate::support::zero_as_none")]
	pub audio_tokens: Option<i32>,
}

impl OutputTokensDetails {
	/// True if all fields are `None`.
	pub fn is_empty(&self) -> bool {
		self.accepted_prediction_tokens.is_none()
			&& self.rejected_prediction_tokens.is_none()
			&& self.reasoning_tokens.is_none()
			&& self.audio_tokens.is_none()
	}
}

// region:    --- Intos

impl From<InputTokensDetails> for PromptTokensDetails {
	fn from(value: InputTokensDetails) -> Self {
		PromptTokensDetails {
			cache_creation_tokens: value.cache_write_tokens,
			cache_creation_details: None,
			cached_tokens: value.cached_tokens,
			audio_tokens: value.audio_tokens,
		}
	}
}

impl From<OutputTokensDetails> for CompletionTokensDetails {
	fn from(value: OutputTokensDetails) -> Self {
		CompletionTokensDetails {
			accepted_prediction_tokens: value.accepted_prediction_tokens,
			rejected_prediction_tokens: value.rejected_prediction_tokens,
			reasoning_tokens: value.reasoning_tokens,
			audio_tokens: value.audio_tokens,
		}
	}
}

impl From<RespUsage> for Usage {
	fn from(value: RespUsage) -> Self {
		Usage {
			prompt_tokens: value.input_tokens,
			prompt_tokens_details: value.input_tokens_details.map(Into::into),
			completion_tokens: value.output_tokens,
			completion_tokens_details: value.output_tokens_details.map(Into::into),
			total_tokens: value.total_tokens,
		}
	}
}

// endregion: --- Intos

// region:    --- Tests

#[cfg(test)]
mod tests {
	use super::RespUsage;
	use crate::chat::Usage;
	use serde_json::json;

	#[test]
	fn test_resp_usage_deserializes_openai_cache_write_tokens() -> Result<(), serde_json::Error> {
		let usage: RespUsage = serde_json::from_value(json!({
			"input_tokens_details": {
				"cache_write_tokens": 77
			}
		}))?;
		let normalized: Usage = usage.into();

		assert_eq!(
			normalized
				.prompt_tokens_details
				.as_ref()
				.and_then(|details| details.cache_creation_tokens),
			Some(77)
		);

		Ok(())
	}
	#[test]
	fn cache_counter_presence_survives_responses_usage_normalization() -> Result<(), serde_json::Error> {
		for (details, write, read) in [
			(json!({}), None, None),
			(json!({"cached_tokens": 0}), None, Some(0)),
			(json!({"cache_write_tokens": 0}), Some(0), None),
			(
				json!({"cache_write_tokens": 17, "cached_tokens": 29}),
				Some(17),
				Some(29),
			),
		] {
			let mut usage: RespUsage = serde_json::from_value(json!({"input_tokens_details": details}))?;
			usage.compact_details();
			let mut normalized: Usage = usage.into();
			normalized.compact_details();
			assert_eq!(
				normalized.prompt_tokens_details.as_ref().and_then(|d| d.cache_creation_tokens),
				write
			);
			assert_eq!(
				normalized.prompt_tokens_details.as_ref().and_then(|d| d.cached_tokens),
				read
			);
			assert_eq!(
				normalized.prompt_tokens_details.is_some(),
				write.is_some() || read.is_some()
			);
			assert_eq!(
				serde_json::from_value::<Usage>(serde_json::to_value(&normalized)?)?,
				normalized
			);
		}
		Ok(())
	}
}

// endregion: --- Tests
