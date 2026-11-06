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
import java.util.UUID;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;

/** Encrypted on-device journal, conversations and project links (schema 2). */
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
        if (store.containsAlias(KEY_ALIAS)) return (SecretKey) store.getKey(KEY_ALIAS, null);
        KeyGenerator generator = KeyGenerator.getInstance("AES", "AndroidKeyStore");
        generator.init(new KeyGenParameterSpec.Builder(KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256).build());
        return generator.generateKey();
    }

    private JSONObject empty() throws Exception {
        JSONObject data = new JSONObject();
        data.put("schema", 2);
        data.put("active_conversation_id", "");
        data.put("conversations", new JSONArray());
        data.put("projects", new JSONArray());
        return data;
    }

    private JSONObject read() throws Exception {
        if (!file.getBaseFile().exists()) return empty();
        try (InputStream input = file.openRead(); ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] block = new byte[4096];
            int n;
            while ((n = input.read(block)) != -1) {
                if (output.size() + n > MAX_FILE_BYTES) throw new IllegalStateException("Encrypted journal size limit reached");
                output.write(block, 0, n);
            }
            byte[] raw = output.toByteArray();
            if (raw.length < IV_BYTES + 16) throw new IllegalStateException("Invalid journal ciphertext");
            byte[] iv = new byte[IV_BYTES];
            System.arraycopy(raw, 0, iv, 0, IV_BYTES);
            Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
            cipher.init(Cipher.DECRYPT_MODE, key(), new GCMParameterSpec(128, iv));
            JSONObject data = new JSONObject(new String(cipher.doFinal(raw, IV_BYTES, raw.length - IV_BYTES), StandardCharsets.UTF_8));
            int schema = data.optInt("schema", -1);
            if (schema == 1) return migrate(data);
            if (schema != 2 || data.optJSONArray("conversations") == null || data.optJSONArray("projects") == null)
                throw new IllegalStateException("Unsupported conversation schema");
            return data;
        }
    }

    private JSONObject migrate(JSONObject old) throws Exception {
        JSONObject data = empty();
        JSONObject conversation = new JSONObject();
        String id = UUID.randomUUID().toString();
        conversation.put("id", id);
        conversation.put("title", "Earlier conversation");
        long now = System.currentTimeMillis();
        conversation.put("created_at", now);
        conversation.put("updated_at", now);
        conversation.put("entries", old.optJSONArray("entries") == null ? new JSONArray() : old.getJSONArray("entries"));
        data.getJSONArray("conversations").put(conversation);
        data.put("active_conversation_id", id);
        write(data);
        return data;
    }

    private void write(JSONObject data) throws Exception {
        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
        cipher.init(Cipher.ENCRYPT_MODE, key());
        byte[] iv = cipher.getIV();
        if (iv.length != IV_BYTES) throw new IllegalStateException("Unexpected IV size");
        byte[] encrypted = cipher.doFinal(data.toString().getBytes(StandardCharsets.UTF_8));
        if (iv.length + encrypted.length > MAX_FILE_BYTES) throw new IllegalStateException("Journal full. Clear history to make room.");
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

    private JSONObject ensureConversation(JSONObject data) throws Exception {
        JSONArray all = data.getJSONArray("conversations");
        String activeId = data.optString("active_conversation_id", "");
        for (int i = 0; i < all.length(); i++) {
            JSONObject item = all.optJSONObject(i);
            if (item != null && activeId.equals(item.optString("id"))) return item;
        }
        JSONObject item = new JSONObject();
        String id = UUID.randomUUID().toString();
        long now = System.currentTimeMillis();
        item.put("id", id);
        item.put("title", "New conversation");
        item.put("created_at", now);
        item.put("updated_at", now);
        item.put("entries", new JSONArray());
        item.put("project_ids", new JSONArray());
        all.put(item);
        data.put("active_conversation_id", id);
        return item;
    }

    public synchronized JSONArray entries() throws Exception {
        JSONObject data = read();
        JSONObject active = ensureConversation(data);
        write(data);
        return new JSONArray(active.getJSONArray("entries").toString());
    }

    public synchronized JSONArray conversations() throws Exception {
        JSONObject data = read();
        ensureConversation(data);
        write(data);
        return new JSONArray(data.getJSONArray("conversations").toString());
    }

    public synchronized JSONArray projects() throws Exception {
        return new JSONArray(read().getJSONArray("projects").toString());
    }

    public synchronized String activeConversationId() throws Exception {
        JSONObject data = read();
        JSONObject active = ensureConversation(data);
        write(data);
        return active.getString("id");
    }

    public synchronized void activateConversation(String id) throws Exception {
        JSONObject data = read();
        JSONArray all = data.getJSONArray("conversations");
        for (int i = 0; i < all.length(); i++) {
            JSONObject item = all.optJSONObject(i);
            if (item != null && id.equals(item.optString("id"))) {
                data.put("active_conversation_id", id);
                write(data);
                return;
            }
        }
        throw new IllegalArgumentException("Conversation not found");
    }

    public synchronized String createConversation() throws Exception {
        JSONObject data = read();
        JSONObject item = new JSONObject();
        String id = UUID.randomUUID().toString();
        long now = System.currentTimeMillis();
        item.put("id", id);
        item.put("title", "New conversation");
        item.put("created_at", now);
        item.put("updated_at", now);
        item.put("entries", new JSONArray());
        item.put("project_ids", new JSONArray());
        data.getJSONArray("conversations").put(item);
        data.put("active_conversation_id", id);
        write(data);
        return id;
    }

    public synchronized String createProject(String title) throws Exception {
        String clean = title == null ? "" : title.trim();
        if (clean.isEmpty() || clean.length() > 60) throw new IllegalArgumentException("Project name must be 1–60 characters");
        JSONObject data = read();
        JSONObject project = new JSONObject();
        project.put("id", UUID.randomUUID().toString());
        project.put("title", clean);
        project.put("created_at", System.currentTimeMillis());
        data.getJSONArray("projects").put(project);
        write(data);
        return project.getString("id");
    }

    public synchronized void linkActiveConversationToProject(String projectId) throws Exception {
        JSONObject data = read();
        JSONObject active = ensureConversation(data);
        linkConversation(data, active, projectId);
        write(data);
    }

    public synchronized void linkConversationToProject(String conversationId, String projectId) throws Exception {
        JSONObject data = read();
        JSONObject conversation = null;
        JSONArray all = data.getJSONArray("conversations");
        for (int i = 0; i < all.length(); i++) {
            JSONObject item = all.optJSONObject(i);
            if (item != null && conversationId.equals(item.optString("id"))) { conversation = item; break; }
        }
        if (conversation == null) throw new IllegalArgumentException("Conversation not found");
        linkConversation(data, conversation, projectId);
        write(data);
    }

    private void linkConversation(JSONObject data, JSONObject active, String projectId) throws Exception {
        boolean exists = false;
        JSONArray projects = data.getJSONArray("projects");
        for (int i = 0; i < projects.length(); i++) {
            JSONObject project = projects.optJSONObject(i);
            if (project != null && projectId.equals(project.optString("id"))) exists = true;
        }
        if (!exists) throw new IllegalArgumentException("Project not found");
        JSONArray links = active.optJSONArray("project_ids");
        if (links == null) { links = new JSONArray(); active.put("project_ids", links); }
        for (int i = 0; i < links.length(); i++) if (projectId.equals(links.optString(i))) return;
        links.put(projectId);
    }

    public synchronized void append(String role, String message) throws Exception {
        if (!"user".equals(role) && !"forge".equals(role) && !"teacher".equals(role)) throw new IllegalArgumentException("Unknown conversation role");
        if (message == null || message.trim().isEmpty() || message.length() > MAX_MESSAGE_CHARS) throw new IllegalArgumentException("Message empty or too long");
        JSONObject data = read();
        JSONObject active = ensureConversation(data);
        long now = System.currentTimeMillis();
        JSONObject item = new JSONObject();
        item.put("role", role);
        item.put("text", message.trim());
        item.put("created_at", now);
        active.getJSONArray("entries").put(item);
        active.put("updated_at", now);
        if ("user".equals(role) && active.getJSONArray("entries").length() == 1)
            active.put("title", message.trim().replaceAll("\\s+", " ").substring(0, Math.min(48, message.trim().replaceAll("\\s+", " ").length())));
        write(data);
    }

    public synchronized int entryCount() throws Exception {
        int count = 0;
        JSONArray all = conversations();
        for (int i = 0; i < all.length(); i++) {
            JSONObject item = all.optJSONObject(i);
            if (item != null) count += item.optJSONArray("entries") == null ? 0 : item.getJSONArray("entries").length();
        }
        return count;
    }

    public synchronized void clear() throws Exception {
        JSONObject data = read();
        JSONArray all = data.getJSONArray("conversations");
        for (int i = 0; i < all.length(); i++) {
            JSONObject item = all.optJSONObject(i);
            if (item != null) {
                item.put("entries", new JSONArray());
                item.put("title", "New conversation");
            }
        }
        write(data);
    }
}
