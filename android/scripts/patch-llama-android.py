#!/usr/bin/env python3
"""Apply the reviewed Forge mobile chat fixes to pinned llama.cpp Android code."""
from pathlib import Path

target = Path("vendor/llama.cpp/examples/llama.android/lib/src/main/cpp/ai_chat.cpp")
source = target.read_text(encoding="utf-8")

def replace_once(old: str, new: str, label: str) -> None:
    global source
    found = source.count(old)
    if found != 1:
        raise RuntimeError(f"{label}: expected exactly one source match, found {found}")
    source = source.replace(old, new, 1)

replace_once(
    "    stop_generation_position = current_position + user_prompt_size + n_predict;",
    """
    // Qwen chat templates may enter a long internal-thinking phase. When the
    // template supports it, use the documented empty thinking-block prefill.
    const auto template_source = common_chat_templates_source(g_chat_templates.get());
    const bool has_qwen_thinking = template_source.find("enable_thinking") != std::string::npos
            && template_source.find("<think>") != std::string::npos;
    if (has_qwen_thinking) {
        const std::string no_think_prefix = "<think>\\n\\n</think>\\n\\n";
        const auto prefill_tokens = common_tokenize(g_context, no_think_prefix, false, true);
        if (decode_tokens_in_batches(g_context, g_batch, prefill_tokens, current_position, true)) {
            LOGe("%s: cannot apply Qwen non-thinking prefix", __func__);
            return 2;
        }
        current_position += (int) prefill_tokens.size();
        assistant_ss << no_think_prefix;
    }
    stop_generation_position = current_position + std::max(1, (int) n_predict);""",
    "thinking-control prefix and output budget")

replace_once(
    '''        LOGw("%s: STOP: hitting stop position: %d", __func__, stop_generation_position);
        return nullptr;''',
    '''        LOGw("%s: STOP: hitting stop position: %d", __func__, stop_generation_position);
        // Close the assistant turn when the output budget is reached.
        chat_add_and_format(ROLE_ASSISTANT, assistant_ss.str());
        return nullptr;''',
    "close assistant turn at output limit")

target.write_text(source, encoding="utf-8")
assert source.count('chat_add_and_format(ROLE_ASSISTANT, assistant_ss.str());') == 2
assert 'stop_generation_position = current_position + std::max(1, (int) n_predict);' in source
print("Forge Core Android llama.cpp patch applied.")
