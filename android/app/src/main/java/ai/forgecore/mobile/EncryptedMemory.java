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
import java.util.LinkedHashSet;
import java.util.Set;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;

/**
 * Android-only encrypted store. Keys stay in Android Keystore; no cloud, no backup.
 * Prototype: not an independently audited storage or recovery implementation.
 */
public final class EncryptedMemory {
    private static final String ALIAS = "forge_core_private_v1";
    private static final int IV_BYTES = 12;
    private static final int MAX_BYTES = 2 * 1024 * 1024;
    private final AtomicFile file;

    public EncryptedMemory(Context context) {
        file = new AtomicFile(new File(context.getFilesDir(), "forge-core-memory.enc"));
    }

    private SecretKey key() throws Exception {
        KeyStore keys = KeyStore.getInstance("AndroidKeyStore");
        keys.load(null);
        if (keys.containsAlias(ALIAS)) {
            return (SecretKey) keys.getKey(ALIAS, null);
        }
        KeyGenerator generator = KeyGenerator.getInstance("AES", "AndroidKeyStore");
        generator.init(new KeyGenParameterSpec.Builder(
                ALIAS, KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build());
        return generator.generateKey();
    }

    private static JSONObject empty() throws Exception {
        JSONObject data = new JSONObject();
        data.put("schema", 1);
        data.put("allowed", new JSONArray());
        data.put("principles", new JSONArray());
        return data;
    }

    private synchronized JSONObject read() throws Exception {
        if (!file.getBaseFile().exists()) return empty();
        try (InputStream in = file.openRead(); ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[4096];
            int length;
            while ((length = in.read(buffer)) != -1) {
                if (out.size() + length > MAX_BYTES) throw new IllegalStateException("Memory limit exceeded");
                out.write(buffer, 0, length);
            }
            byte[] raw = out.toByteArray();
            if (raw.length < IV_BYTES + 16) throw new IllegalStateException("Invalid memory file");
            byte[] iv = new byte[IV_BYTES];
            System.arraycopy(raw, 0, iv, 0, IV_BYTES);
            Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
            cipher.init(Cipher.DECRYPT_MODE, key(), new GCMParameterSpec(128, iv));
            byte[] text = cipher.doFinal(raw, IV_BYTES, raw.length - IV_BYTES);
            JSONObject data = new JSONObject(new String(text, StandardCharsets.UTF_8));
            if (data.optInt("schema") != 1) throw new IllegalStateException("Unsupported memory schema");
            return data;
        }
    }

    private synchronized void write(JSONObject data) throws Exception {
        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
        cipher.init(Cipher.ENCRYPT_MODE, key());
        byte[] iv = cipher.getIV();
        if (iv.length != IV_BYTES) throw new IllegalStateException("Unexpected cipher IV");
        byte[] encrypted = cipher.doFinal(data.toString().getBytes(StandardCharsets.UTF_8));
        if (iv.length + encrypted.length > MAX_BYTES) throw new IllegalStateException("Memory limit exceeded");
        FileOutputStream out = null;
        try {
            out = file.startWrite();
            out.write(iv);
            out.write(encrypted);
            file.finishWrite(out);
        } catch (Exception error) {
            if (out != null) file.failWrite(out);
            throw error;
        }
    }

    public synchronized Set<String> allowed() throws Exception {
        JSONArray values = read().optJSONArray("allowed");
        Set<String> packages = new LinkedHashSet<>();
        if (values != null) {
            for (int i = 0; i < values.length(); i++) {
                String name = values.optString(i, "");
                if (!name.isEmpty()) packages.add(name);
            }
        }
        return packages;
    }

    public synchronized void setAllowed(Set<String> packages) throws Exception {
        JSONObject data = read();
        JSONArray values = new JSONArray();
        for (String name : packages) values.put(name);
        data.put("allowed", values);
        write(data);
    }

    public synchronized void remember(String packageName, String reason, String principle) throws Exception {
        if (packageName.trim().isEmpty() || reason.trim().isEmpty() || principle.trim().isEmpty()) {
            throw new IllegalArgumentException("All fields must be present");
        }
        JSONObject data = read();
        JSONObject record = new JSONObject();
        record.put("app", packageName);
        record.put("why", reason.trim());
        record.put("principle", principle.trim());
        record.put("confirmed_at", System.currentTimeMillis());
        JSONArray principles = data.getJSONArray("principles");
        principles.put(record);
        write(data);
    }

    public synchronized JSONArray principles() throws Exception {
        return read().getJSONArray("principles");
    }

    public synchronized void forgetPrinciples() throws Exception {
        JSONObject data = read();
        data.put("principles", new JSONArray());
        write(data);
    }
}
