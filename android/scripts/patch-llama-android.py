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

# __android_log_is_loggable is only available from Android 11 / API 30.
# Forge Core keeps its existing API 26 minimum, so use the configured priority
# threshold on older Android targets.
logging_target = Path("vendor/llama.cpp/examples/llama.android/lib/src/main/cpp/logging.h")
logging_source = logging_target.read_text(encoding="utf-8")
old_logging = "    return __android_log_is_loggable(prio, LOG_TAG, LOG_MIN_LEVEL);"
new_logging = """#if __ANDROID_API__ >= 30
    return __android_log_is_loggable(prio, LOG_TAG, LOG_MIN_LEVEL);
#else
    return prio >= LOG_MIN_LEVEL;
#endif"""
if logging_source.count(old_logging) != 1:
    raise RuntimeError("Android API 26 logging fallback: expected one source match")
logging_source = logging_source.replace(old_logging, new_logging, 1)
logging_target.write_text(logging_source, encoding="utf-8")
assert "#if __ANDROID_API__ >= 30" in logging_source

# The upstream Android wrapper maps every nonzero native model-load result to
# UnsupportedArchitectureException, even for memory pressure or malformed
# tensors. Preserve the llama.cpp warning/error lines so Forge can report the
# actual loader failure instead of misdiagnosing every failure as architecture.
old_includes = "#include <android/log.h>"
new_includes = """#include <android/log.h>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>

static std::mutex aichat_model_load_log_mutex;
static std::string aichat_model_load_log;
static bool aichat_capture_model_load_logs = false;"""
if logging_source.count(old_includes) != 1:
    raise RuntimeError("Native diagnostic includes: expected one android/log include")
logging_source = logging_source.replace(old_includes, new_includes, 1)
old_callback = """    const int prio = android_log_prio_from_ggml(level);
    if (!ai_should_log(prio)) return;
    __android_log_write(prio, LOG_TAG, text);"""
new_callback = """    const int prio = android_log_prio_from_ggml(level);
    if (text && level <= GGML_LOG_LEVEL_WARN) {
        std::lock_guard<std::mutex> lock(aichat_model_load_log_mutex);
        if (aichat_capture_model_load_logs && aichat_model_load_log.size() < 8192) {
            const size_t remaining = 8192 - aichat_model_load_log.size();
            aichat_model_load_log.append(text, std::min(std::strlen(text), remaining));
        }
    }
    if (!ai_should_log(prio)) return;
    __android_log_write(prio, LOG_TAG, text);"""
if logging_source.count(old_callback) != 1:
    raise RuntimeError("Native diagnostic callback: expected one Android logger")
logging_source = logging_source.replace(old_callback, new_callback, 1)
logging_target.write_text(logging_source, encoding="utf-8")
assert "aichat_model_load_log.append" in logging_source

cpp_target = Path("vendor/llama.cpp/examples/llama.android/lib/src/main/cpp/ai_chat.cpp")
cpp_source = cpp_target.read_text(encoding="utf-8")
old_load = """    auto *model = llama_model_load_from_file(model_path, model_params);
    env->ReleaseStringUTFChars(jmodel_path, model_path);
    if (!model) {
        return 1;
    }"""
new_load = """    {
        std::lock_guard<std::mutex> lock(aichat_model_load_log_mutex);
        aichat_model_load_log.clear();
        aichat_capture_model_load_logs = true;
    }
    auto *model = llama_model_load_from_file(model_path, model_params);
    {
        std::lock_guard<std::mutex> lock(aichat_model_load_log_mutex);
        aichat_capture_model_load_logs = false;
    }
    env->ReleaseStringUTFChars(jmodel_path, model_path);
    if (!model) {
        return 1;
    }"""
if cpp_source.count(old_load) != 1:
    raise RuntimeError("Native model load capture: expected one JNI loader block")
cpp_source = cpp_source.replace(old_load, new_load, 1)
getter_anchor = """    g_model = model;
    return 0;
}

static llama_context *init_context"""
getter_replacement = """    g_model = model;
    return 0;
}

extern \"C\"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_lastModelLoadLog(JNIEnv *env, jobject /*unused*/) {
    std::lock_guard<std::mutex> lock(aichat_model_load_log_mutex);
    return env->NewStringUTF(aichat_model_load_log.c_str());
}

static llama_context *init_context"""
if cpp_source.count(getter_anchor) != 1:
    raise RuntimeError("Native model log getter: expected one post-load anchor")
cpp_source = cpp_source.replace(getter_anchor, getter_replacement, 1)
cpp_target.write_text(cpp_source, encoding="utf-8")
assert "InferenceEngineImpl_lastModelLoadLog" in cpp_source

kotlin_target = Path("vendor/llama.cpp/examples/llama.android/lib/src/main/java/com/arm/aichat/internal/InferenceEngineImpl.kt")
kotlin_source = kotlin_target.read_text(encoding="utf-8")
native_method_anchor = """    @FastNative
    private external fun load(modelPath: String): Int

    @FastNative
    private external fun prepare(): Int"""
native_method_replacement = """    @FastNative
    private external fun load(modelPath: String): Int

    @FastNative
    private external fun lastModelLoadLog(): String

    @FastNative
    private external fun prepare(): Int"""
if kotlin_source.count(native_method_anchor) != 1:
    raise RuntimeError("Kotlin native diagnostic method: expected one load/prepare declaration")
kotlin_source = kotlin_source.replace(native_method_anchor, native_method_replacement, 1)
old_exception = """                    if (it != 0) throw UnsupportedArchitectureException()"""
new_exception = """                    if (it != 0) {
                        val detail = lastModelLoadLog().trim().takeLast(1200)
                        val suffix = if (detail.isBlank()) "No native warning or error was captured." else detail
                        throw IllegalStateException("GGUF loader failed (native code $it): $suffix")
                    }"""
if kotlin_source.count(old_exception) != 1:
    raise RuntimeError("Kotlin native diagnostic error: expected generic architecture mapping")
kotlin_source = kotlin_source.replace(old_exception, new_exception, 1)
kotlin_target.write_text(kotlin_source, encoding="utf-8")
assert "GGUF loader failed (native code" in kotlin_source
print("Forge Core Android llama.cpp patch applied.")
