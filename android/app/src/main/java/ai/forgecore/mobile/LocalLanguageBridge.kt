package ai.forgecore.mobile

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.security.MessageDigest
import java.nio.file.Files
import java.nio.file.StandardCopyOption

/**
 * Optional offline bridge powered by a model the user imports locally.
 * It is explicitly distinct from Forge Seed and never changes training data.
 */
class LocalLanguageBridge(context: Context) : ForgeModelAdapter {
    private val app = context.applicationContext
    private val engine = AiChat.getInferenceEngine(app)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val preferences = app.getSharedPreferences("forge_bridge_model", Context.MODE_PRIVATE)
    private val modelFile = File(File(app.filesDir, "models"), "language-bridge.gguf")
    private val maxModelBytes = 3_000_000_000L
    @Volatile private var importedName = preferences.getString("name", "") ?: ""
    @Volatile private var importedHash = preferences.getString("sha256", "") ?: ""
    @Volatile private var operation = "No local language model selected."
    @Volatile private var loading = false
    @Volatile private var importing = false
    @Volatile private var generation: Job? = null
    @Volatile private var modelLoaded = engine.state.value is InferenceEngine.State.ModelReady ||
        engine.state.value is InferenceEngine.State.Generating

    override fun status(): ForgeModelAdapter.ModelStatus {
        val state = engine.state.value
        val ready = modelLoaded || state is InferenceEngine.State.ModelReady ||
            state is InferenceEngine.State.Generating
        val hasImportedFile = modelFile.isFile
        val name = if (hasImportedFile && importedName.isNotBlank()) importedName else "No local model selected"
        val provenance = if (hasImportedFile && importedHash.isNotBlank()) {
            "User-imported pretrained GGUF • SHA-256 ${importedHash.take(12)}…"
        } else "No model weights imported"
        val detail = when {
            ready -> "Optional local language bridge ready. This is not Forge Seed."
            loading -> "Loading the local language bridge; this may take a few minutes."
            importing -> operation
            modelFile.isFile -> "Model is stored locally. Tap Load when you want to start a chat."
            else -> operation
        }
        return ForgeModelAdapter.ModelStatus(
            ready,
            if (ready) "Local language bridge" else name,
            "bridge-v1",
            provenance,
            detail
        )
    }

    fun hasImportedModelFile(): Boolean = modelFile.isFile
    fun isBusy(): Boolean = loading || importing

    fun importModel(uri: Uri, listener: ForgeModelAdapter.OperationListener) {
        val state = engine.state.value
        if (loading || importing || generation?.isActive == true ||
            state is InferenceEngine.State.ModelReady || state is InferenceEngine.State.Generating ||
            state is InferenceEngine.State.LoadingModel || state is InferenceEngine.State.ProcessingSystemPrompt ||
            state is InferenceEngine.State.ProcessingUserPrompt) {
            listener.onError("Unload or finish the current model session before importing another model.")
            return
        }
        scope.launch {
            importing = true
            val temporary = File(modelFile.parentFile, "language-bridge.incoming.part")
            try {
                operation = "Copying and checking the selected GGUF…"
                listener.onProgress(operation)
                val result = kotlinx.coroutines.withContext(Dispatchers.IO) {
                    modelFile.parentFile?.mkdirs()
                    val resolver = app.contentResolver
                    val name = resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                        ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
                        ?: "local-model.gguf"
                    require(name.lowercase().endsWith(".gguf")) { "Choose a .gguf model file." }
                    val digest = MessageDigest.getInstance("SHA-256")
                    var count = 0L
                    var lastProgressBytes = 0L
                    val input = resolver.openInputStream(uri)
                        ?: throw IllegalArgumentException("Android could not open the selected file.")
                    input.use { source ->
                        FileOutputStream(temporary).use { output ->
                            val buffer = ByteArray(256 * 1024)
                            while (true) {
                                val n = source.read(buffer)
                                if (n < 0) break
                                count += n
                                require(count <= maxModelBytes) { "Model exceeds the 3 GB import limit." }
                                output.write(buffer, 0, n)
                                digest.update(buffer, 0, n)
                                if (count - lastProgressBytes >= 32L * 1024 * 1024) {
                                    lastProgressBytes = count
                                    val progress = "Copied ${count / (1024 * 1024)} MB; checking GGUF and SHA-256…"
                                    operation = progress
                                    listener.onProgress(progress)
                                }
                            }
                            output.fd.sync()
                        }
                    }
                    require(count >= 64L) { "Selected file is too small to be a GGUF model." }
                    FileInputStream(temporary).use { inputFile ->
                        val header = ByteArray(4)
                        require(inputFile.read(header) == 4 && String(header, Charsets.US_ASCII) == "GGUF") {
                            "Selected file does not have a GGUF header."
                        }
                    }
                    val hash = digest.digest().joinToString("") { "%02x".format(it) }
                    try {
                        Files.move(temporary.toPath(), modelFile.toPath(),
                            StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
                    } catch (_: Exception) {
                        Files.move(temporary.toPath(), modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING)
                    }
                    Triple(name, hash, count)
                }
                importedName = result.first
                importedHash = result.second
                preferences.edit().putString("name", importedName)
                    .putString("sha256", importedHash).putLong("bytes", result.third).apply()
                operation = "Model imported and checked. Loading is a separate action."
                listener.onComplete(operation)
            } catch (error: Exception) {
                temporary.delete()
                operation = "Model import failed: ${error.message ?: "unknown error"}"
                listener.onError(operation)
            } finally {
                importing = false
            }
        }
    }

    fun loadModel(listener: ForgeModelAdapter.OperationListener) {
        val state = engine.state.value
        if (loading || importing || generation?.isActive == true || state is InferenceEngine.State.ModelReady) {
            listener.onError("The model is already loading, generating, or loaded.")
            return
        }
        if (!modelFile.isFile || modelFile.length() < 64L) {
            listener.onError("Choose a local GGUF model first.")
            return
        }
        scope.launch {
            loading = true
            operation = "Loading local model into memory…"
            listener.onProgress("Loading local model into phone memory. This can take a few minutes.")
            try {
                val priorState = engine.state.first {
                    it is InferenceEngine.State.Initialized || it is InferenceEngine.State.Error ||
                        it is InferenceEngine.State.ModelReady
                }
                if (priorState is InferenceEngine.State.ModelReady) {
                    modelLoaded = true
                    operation = "Local language bridge ready. Replies are generated on this device."
                    listener.onComplete(operation)
                    return@launch
                }
                if (priorState is InferenceEngine.State.Error) engine.cleanUp()
                engine.loadModel(modelFile.absolutePath)
                engine.setSystemPrompt(
                    "You are an optional local language bridge in the Forge Core app. " +
                    "You are a separate pretrained model, not Forge Seed, and you do not train Forge. " +
                    "Never claim to be the scratch-trained Forge model. Be candid about uncertainty. " +
                    "Help the user think through ideas and ask useful, respectful follow-up questions. " +
                    "Only the user can confirm or save personal principles. /no_think"
                )
                modelLoaded = true
                operation = "Local language bridge ready. Replies are generated on this device."
                listener.onComplete(operation)
            } catch (error: Exception) {
                modelLoaded = false
                operation = "Model load failed: ${error.message ?: "unsupported GGUF or insufficient memory"}"
                listener.onError(operation)
            } finally {
                loading = false
            }
        }
    }

    fun unloadModel(listener: ForgeModelAdapter.OperationListener) {
        if (loading || generation?.isActive == true) {
            listener.onError("Wait for loading or generation to finish first.")
            return
        }
        if (engine.state.value !is InferenceEngine.State.ModelReady) {
            listener.onError("No model is currently loaded.")
            return
        }
        scope.launch {
            try {
                engine.cleanUp()
                modelLoaded = false
                operation = "Model unloaded from memory. The local model file remains on this device."
                listener.onComplete(operation)
            } catch (error: Exception) {
                modelLoaded = false
                operation = "Could not unload model: ${error.message ?: "unknown error"}"
                listener.onError(operation)
            }
        }
    }

    fun removeModelFile(listener: ForgeModelAdapter.OperationListener) {
        val state = engine.state.value
        if (loading || importing || generation?.isActive == true || state is InferenceEngine.State.ModelReady ||
            state is InferenceEngine.State.Generating || state is InferenceEngine.State.LoadingModel) {
            listener.onError("Unload the model and finish any response before removing its file.")
            return
        }
        scope.launch {
            if (!modelFile.exists() || modelFile.delete()) {
                preferences.edit().remove("name").remove("sha256").remove("bytes").apply()
                importedName = ""
                importedHash = ""
                operation = "Imported model file removed from this device."
                listener.onComplete(operation)
            } else {
                operation = "Could not remove the imported model file."
                listener.onError(operation)
            }
        }
    }

    override fun generate(prompt: String, listener: ForgeModelAdapter.Listener) {
        if (engine.state.value !is InferenceEngine.State.ModelReady) {
            listener.onError("Load the local language bridge first.")
            return
        }
        if (prompt.length > 5_000) {
            listener.onError("Message is longer than the 5,000-character local chat limit.")
            return
        }
        generation = scope.launch {
            try {
                engine.sendUserPrompt(prompt, predictLength = 256).collect { token ->
                    listener.onToken(token)
                }
                listener.onComplete()
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                if (engine.state.value is InferenceEngine.State.Error) modelLoaded = false
                listener.onError(error.message ?: "Local inference failed.")
            } finally {
                generation = null
            }
        }
    }

    override fun cancel() {
        generation?.cancel()
    }
}
