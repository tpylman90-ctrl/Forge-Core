package ai.forgecore.mobile;

/**
 * Local Forge-native inference boundary. No remote requests or pretrained fallback.
 *
 * A real implementation must identify its weights and provenance, stay offline,
 * deliver callbacks without blocking the UI, and implement cooperative cancellation.
 * The Android shell never treats a missing model as a conversational agent.
 */
public interface ForgeModelAdapter {
    final class ModelStatus {
        public final boolean available;
        public final String name;
        public final String version;
        public final String provenance;
        public final String detail;

        public ModelStatus(boolean available, String name, String version,
                           String provenance, String detail) {
            this.available = available;
            this.name = name;
            this.version = version;
            this.provenance = provenance;
            this.detail = detail;
        }
    }

    interface Listener {
        void onToken(String token);
        void onComplete();
        void onError(String reason);
    }

    ModelStatus status();
    void generate(String prompt, Listener listener);
    void cancel();

    /** Honest default until Forge-owned, tested Android inference exists. */
    final class Unavailable implements ForgeModelAdapter {
        @Override public ModelStatus status() {
            return new ModelStatus(false, "No model installed", "none", "none",
                    "Forge Seed training is separate; this app cannot generate replies.");
        }

        @Override public void generate(String prompt, Listener listener) {
            listener.onError("Forge-native inference is not installed.");
        }

        @Override public void cancel() {
            // No inference is running.
        }
    }
}
