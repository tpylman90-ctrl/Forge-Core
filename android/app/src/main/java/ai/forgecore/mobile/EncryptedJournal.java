package ai.forgecore.mobile;

import android.content.Context;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;
import android.util.AtomicFile;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.KeyStore;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;

/**
 * Independent, encrypted on-device conversation journal (schema 1).
 * Never shares a file or key alias with the v0.2 confirmed-principles vault.
 * Journal entries are not promoted to learned principles automatically.
 */
public final class EncryptedJournal {
    private static final String KEY_ALIAS = "forge_core_journal_v1";
    private static final int IV_BYTES = 12;
    private static final int MAX_FILE_BYTES = 1024 * 1024;
    public static final int MAX_MESSAGE_CHARS = 8000;
    private final AtomicFile file;

    public EncryptedJournal(Context context) {
        file = new AtomicFile(new File(context.getFilesDir(), "forge-core-journal.enc"));
    }

    private SecretKey key() throws Exception {
        KeyStore store = KeyStore.getInstance("AndroidKeyStore");
        store.load(null);
        if (store.containsAlias(KEY_ALIAS)) {
            return (SecretKey) store.getKey(KEY_ALIAS, null);
        }
        KeyGenerator generator = KeyGenerator.getInstance("AES", "AndroidKeyStore");
        generator.init(new KeyGenParameterSpec.Builder(KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build());
        return generator.generateKey();
    }

    private JSONObject empty() throws Exception {
        JSONObject data = new JSONObject();
        data.put("schema", 1);
        data.put("entries", new JSONArray());
        return data;
    }

    private synchronized JSONObject read() throws Exception {
        if (!file.getBaseFile().exists()) return empty();
        try (InputStream input = file.openRead();
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] block = new byte[4096];
            int n;
            while ((n = input.read(block)) != -1) {
                if (output.size() + n > MAX_FILE_BYTES) {
                    throw new IllegalStateException("Encrypted journal size limit reached");
                }
                output.write(block, 0, n);
            }
            byte[] raw = output.toByteArray();
            if (raw.length < IV_BYTES + 16) {
                throw new IllegalStateException("Invalid journal ciphertext");
            }
            byte[] iv = new byte[IV_BYTES];
            System.arraycopy(raw, 0, iv, 0, IV_BYTES);
            Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
            cipher.init(Cipher.DECRYPT_MODE, key(), new GCMParameterSpec(128, iv));
            byte[] decrypted = cipher.doFinal(raw, IV_BYTES, raw.length - IV_BYTES);
            JSONObject data = new JSONObject(new String(decrypted, StandardCharsets.UTF_8));
            if (data.optInt("schema", -1) != 1 || data.optJSONArray("entries") == null) {
                throw new IllegalStateException("Unsupported conversation schema");
            }
            return data;
        }
    }

    private synchronized void write(JSONObject data) throws Exception {
        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
        cipher.init(Cipher.ENCRYPT_MODE, key());
        byte[] iv = cipher.getIV();
        if (iv.length != IV_BYTES) throw new IllegalStateException("Unexpected IV size");
        byte[] encrypted = cipher.doFinal(data.toString().getBytes(StandardCharsets.UTF_8));
        if (iv.length + encrypted.length > MAX_FILE_BYTES) {
            throw new IllegalStateException("Journal full. Clear history to make room.");
        }
        FileOutputStream output = null;
        try {
            output = file.startWrite();
            output.write(iv);
            output.write(encrypted);
            file.finishWrite(output);
        } catch (Exception error) {
            if (output != null) file.failWrite(output);
            throw error;
        }
    }

    /** Returns a fresh JSON array. Never silently deletes previous entries. */
    public synchronized JSONArray entries() throws Exception {
        return read().getJSONArray("entries");
    }

    public synchronized void append(String role, String message) throws Exception {
        if (!"user".equals(role) && !"forge".equals(role)) {
            throw new IllegalArgumentException("Unknown conversation role");
        }
        if (message == null || message.trim().isEmpty()
                || message.length() > MAX_MESSAGE_CHARS) {
            throw new IllegalArgumentException("Message empty or too long");
        }
        JSONObject data = read();
        JSONObject item = new JSONObject();
        item.put("role", role);
        item.put("text", message.trim());
        item.put("created_at", System.currentTimeMillis());
        data.getJSONArray("entries").put(item);
        write(data);
    }

    public synchronized void clear() throws Exception {
        JSONObject data = read();
        data.put("entries", new JSONArray());
        write(data);
    }
}
