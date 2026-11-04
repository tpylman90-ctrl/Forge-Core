package ai.forgecore.mobile;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.AppOpsManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.os.Process;
import android.text.InputFilter;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.text.DateFormat;
import java.util.Date;

/**
 * Offline-first Forge communication shell.
 * The initial adapter is explicitly unavailable: no generated replies or fake AI.
 * Existing v0.2 teaching, observation and memory vault remain in MainActivity.
 */
public final class ForgeHomeActivity extends Activity {
    private static final int BG = Color.rgb(12, 18, 29);
    private static final int SURFACE = Color.rgb(22, 33, 48);
    private static final int SURFACE_RAISED = Color.rgb(29, 45, 62);
    private static final int ACCENT = Color.rgb(94, 214, 236);
    private static final int WHITE = Color.rgb(236, 245, 251);
    private static final int MUTED = Color.rgb(161, 182, 199);
    private static final String[] TABS = {"Chat", "History", "Memory", "Settings", "Diagnostics"};

    private final ForgeModelAdapter model = new ForgeModelAdapter.Unavailable();
    private EncryptedJournal journal;
    private FrameLayout content;
    private TextView[] navigation;
    private int selectedTab = 0;
    private boolean generating = false;
    private StringBuilder streaming = new StringBuilder();
    private String notice = "";

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        journal = new EncryptedJournal(this);
        getWindow().setStatusBarColor(BG);
        getWindow().setNavigationBarColor(BG);
        getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE);

        LinearLayout root = column();
        root.setBackgroundColor(BG);
        root.setFitsSystemWindows(true);

        LinearLayout header = column();
        header.setPadding(dp(20), dp(18), dp(20), dp(12));
        header.setBackgroundColor(SURFACE);
        TextView wordmark = text("◈   FORGE CORE", 22, WHITE, true);
        wordmark.setLetterSpacing(0.07f);
        header.addView(wordmark);
        TextView strap = text("PRIVATE  /  OFFLINE  /  YOUR CONTROL", 10, ACCENT, true);
        strap.setLetterSpacing(0.12f);
        strap.setPadding(0, dp(5), 0, 0);
        header.addView(strap);
        root.addView(header);

        content = new FrameLayout(this);
        root.addView(content, new LinearLayout.LayoutParams(-1, 0, 1f));

        LinearLayout nav = new LinearLayout(this);
        nav.setOrientation(LinearLayout.HORIZONTAL);
        nav.setGravity(Gravity.CENTER);
        nav.setBackgroundColor(SURFACE);
        nav.setPadding(dp(6), dp(4), dp(6), dp(8));
        navigation = new TextView[TABS.length];
        for (int i = 0; i < TABS.length; i++) {
            final int index = i;
            TextView tab = text(TABS[i], 11, MUTED, i == 0);
            tab.setGravity(Gravity.CENTER);
            tab.setContentDescription("Open " + TABS[i]);
            tab.setOnClickListener(view -> showTab(index));
            LinearLayout.LayoutParams tabParams = new LinearLayout.LayoutParams(0, dp(52), 1f);
            tabParams.setMargins(dp(2), 0, dp(2), 0);
            nav.addView(tab, tabParams);
            navigation[i] = tab;
        }
        root.addView(nav);
        setContentView(root);
        showTab(state == null ? 0 : state.getInt("selectedTab", 0));
    }

    @Override protected void onSaveInstanceState(Bundle output) {
        output.putInt("selectedTab", selectedTab);
        super.onSaveInstanceState(output);
    }

    @Override protected void onResume() {
        super.onResume();
        if (content != null) showTab(selectedTab);
    }

    @Override public void onBackPressed() {
        if (selectedTab != 0) {
            showTab(0);
        } else {
            super.onBackPressed();
        }
    }

    private int dp(float value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private LinearLayout column() {
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        return layout;
    }

    private GradientDrawable shape(int color, int radiusDp) {
        GradientDrawable shape = new GradientDrawable();
        shape.setColor(color);
        shape.setCornerRadius(dp(radiusDp));
        return shape;
    }

    private TextView text(String value, int size, int color, boolean bold) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(size);
        view.setTextColor(color);
        if (bold) view.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        view.setLineSpacing(dp(2), 1f);
        return view;
    }

    private void space(LinearLayout parent, int dpValue) {
        View spacer = new View(this);
        parent.addView(spacer, new LinearLayout.LayoutParams(1, dp(dpValue)));
    }

    private LinearLayout panel(LinearLayout parent, boolean raised) {
        LinearLayout card = column();
        card.setPadding(dp(16), dp(15), dp(16), dp(15));
        card.setBackground(shape(raised ? SURFACE_RAISED : SURFACE, 17));
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, -2);
        params.bottomMargin = dp(12);
        parent.addView(card, params);
        return card;
    }

    private TextView action(String label, Runnable callback, boolean prominent) {
        TextView button = text(label, 14, prominent ? BG : WHITE, true);
        button.setGravity(Gravity.CENTER);
        button.setPadding(dp(14), dp(12), dp(14), dp(12));
        button.setMinHeight(dp(48));
        button.setBackground(shape(prominent ? ACCENT : SURFACE_RAISED, 13));
        button.setOnClickListener(view -> callback.run());
        return button;
    }

    private void addAction(LinearLayout parent, String label, Runnable callback, boolean primary) {
        TextView button = action(label, callback, primary);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, -2);
        params.bottomMargin = dp(10);
        parent.addView(button, params);
    }

    private LinearLayout scrollPage() {
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setClipToPadding(false);
        scroll.setPadding(dp(18), dp(16), dp(18), dp(14));
        LinearLayout body = column();
        scroll.addView(body);
        content.addView(scroll, new FrameLayout.LayoutParams(-1, -1));
        return body;
    }

    private void heading(LinearLayout parent, String title, String subtitle) {
        parent.addView(text(title, 25, WHITE, true));
        space(parent, 5);
        parent.addView(text(subtitle, 13, MUTED, false));
        space(parent, 20);
    }

    private void showTab(int index) {
        selectedTab = Math.max(0, Math.min(index, TABS.length - 1));
        content.removeAllViews();
        for (int i = 0; i < navigation.length; i++) {
            navigation[i].setTextColor(i == selectedTab ? ACCENT : MUTED);
            navigation[i].setTypeface(Typeface.DEFAULT,
                    i == selectedTab ? Typeface.BOLD : Typeface.NORMAL);
            navigation[i].setBackground(shape(i == selectedTab ? SURFACE_RAISED : SURFACE, 12));
        }
        switch (selectedTab) {
            case 0: renderChat(); break;
            case 1: renderHistory(); break;
            case 2: renderMemory(); break;
            case 3: renderSettings(); break;
            default: renderDiagnostics(); break;
        }
    }

    private JSONArray readEntriesOrWarn(LinearLayout destination) {
        try {
            return journal.entries();
        } catch (Exception error) {
            LinearLayout warning = panel(destination, true);
            warning.addView(text("Encrypted journal unavailable", 16, WHITE, true));
            space(warning, 6);
            warning.addView(text("Existing data has NOT been deleted. Avoid clearing app data or reinstalling until the issue is resolved.", 13, MUTED, false));
            return null;
        }
    }

    private void renderChat() {
        LinearLayout screen = column();
        content.addView(screen, new FrameLayout.LayoutParams(-1, -1));

        ScrollView scrolling = new ScrollView(this);
        scrolling.setFillViewport(true);
        LinearLayout.LayoutParams scrollingParams = new LinearLayout.LayoutParams(-1, 0, 1f);
        screen.addView(scrolling, scrollingParams);
        LinearLayout messages = column();
        messages.setPadding(dp(18), dp(18), dp(18), dp(8));
        scrolling.addView(messages);

        LinearLayout status = panel(messages, false);
        status.addView(text("●  MODEL NOT CONNECTED", 12, ACCENT, true));
        space(status, 9);
        status.addView(text("The real Forge language model is still being trained.", 16, WHITE, true));
        space(status, 6);
        status.addView(text("Use this space as an encrypted local conversation journal. Notes are not sent to an AI or promoted into learned memories.", 13, MUTED, false));

        if (!notice.isEmpty()) {
            LinearLayout info = panel(messages, true);
            info.addView(text(notice, 13, ACCENT, false));
        }

        JSONArray entries = readEntriesOrWarn(messages);
        if (entries != null && entries.length() == 0) {
            LinearLayout welcome = panel(messages, true);
            welcome.addView(text("A place for your conversations with Forge", 18, WHITE, true));
            space(welcome, 9);
            welcome.addView(text("Write a thought, question, or idea. Your words stay encrypted on this device until the native Forge model is ready.", 14, MUTED, false));
        } else if (entries != null) {
            for (int i = 0; i < entries.length(); i++) {
                JSONObject entry = entries.optJSONObject(i);
                if (entry != null) addMessage(messages, entry);
            }
        }

        if (generating) {
            LinearLayout live = panel(messages, true);
            live.addView(text("Forge is generating locally…", 13, ACCENT, true));
            if (streaming.length() > 0) {
                space(live, 8);
                live.addView(text(streaming.toString(), 14, WHITE, false));
            }
            addAction(live, "Cancel generation", () -> {
                model.cancel();
                generating = false;
                streaming.setLength(0);
                notice = "Generation canceled. No unfinished reply was saved.";
                showTab(0);
            }, false);
        }

        scrolling.post(() -> scrolling.fullScroll(View.FOCUS_DOWN));

        LinearLayout composer = column();
        composer.setPadding(dp(16), dp(12), dp(16), dp(14));
        composer.setBackgroundColor(SURFACE);
        EditText input = new EditText(this);
        input.setHint("Write to your private journal…");
        input.setHintTextColor(MUTED);
        input.setTextColor(WHITE);
        input.setTextSize(15);
        input.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
                | InputType.TYPE_TEXT_FLAG_MULTI_LINE);
        input.setMinLines(1);
        input.setMaxLines(4);
        input.setFilters(new InputFilter[]{new InputFilter.LengthFilter(EncryptedJournal.MAX_MESSAGE_CHARS)});
        input.setBackground(shape(SURFACE_RAISED, 12));
        input.setPadding(dp(14), dp(8), dp(14), dp(8));
        composer.addView(input, new LinearLayout.LayoutParams(-1, -2));
        space(composer, 9);
        TextView send = action(generating ? "Generating…" : "Save private note", () -> {
            String prompt = input.getText().toString().trim();
            if (prompt.isEmpty()) {
                Toast.makeText(this, "Write a note first", Toast.LENGTH_SHORT).show();
                return;
            }
            submitPrompt(prompt);
        }, true);
        send.setEnabled(!generating);
        send.setAlpha(generating ? 0.55f : 1f);
        composer.addView(send, new LinearLayout.LayoutParams(-1, -2));
        screen.addView(composer);
    }

    private void addMessage(LinearLayout parent, JSONObject item) {
        boolean fromForge = "forge".equals(item.optString("role"));
        LinearLayout bubble = panel(parent, fromForge);
        bubble.addView(text(fromForge ? "FORGE · VERIFIED LOCAL MODEL" : "YOU · PRIVATE NOTE",
                10, fromForge ? ACCENT : MUTED, true));
        space(bubble, 8);
        TextView body = text(item.optString("text"), 15, WHITE, false);
        body.setTextIsSelectable(true);
        bubble.addView(body);
        space(bubble, 9);
        bubble.addView(text(formatTime(item.optLong("created_at")),
                11, MUTED, false));
    }

    private String formatTime(long millis) {
        if (millis <= 0) return "Time unknown";
        return DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT)
                .format(new Date(millis));
    }

    private void submitPrompt(String prompt) {
        if (generating) return;
        try {
            journal.append("user", prompt);
            notice = model.status().available
                    ? "Sent to locally installed Forge model."
                    : "Saved on this device. No model reply was generated.";
        } catch (Exception error) {
            notice = "Could not encrypt and save this message. No data was sent.";
            showTab(0);
            return;
        }

        if (!model.status().available) {
            showTab(0);
            return;
        }

        generating = true;
        streaming = new StringBuilder();
        showTab(0);
        try {
            model.generate(prompt, new ForgeModelAdapter.Listener() {
                @Override public void onToken(String token) {
                    runOnUiThread(() -> {
                        if (!generating) return;
                        if (token != null) streaming.append(token);
                        if (selectedTab == 0) showTab(0);
                    });
                }
                @Override public void onComplete() {
                    runOnUiThread(() -> {
                        if (!generating) return;
                        generating = false;
                        String answer = streaming.toString();
                        streaming.setLength(0);
                        if (!answer.trim().isEmpty()) {
                            try {
                                journal.append("forge", answer);
                                notice = "Local model response saved.";
                            } catch (Exception error) {
                                notice = "Model replied, but secure storage failed; response was not saved.";
                            }
                        } else notice = "Local model finished without a reply.";
                        if (selectedTab == 0) showTab(0);
                    });
                }
                @Override public void onError(String reason) {
                    runOnUiThread(() -> {
                        if (!generating) return;
                        generating = false;
                        streaming.setLength(0);
                        notice = "Local inference unavailable: " + (reason == null ? "unknown error" : reason);
                        if (selectedTab == 0) showTab(0);
                    });
                }
            });
        } catch (Exception error) {
            generating = false;
            streaming.setLength(0);
            notice = "Local inference failed. Your note remains saved.";
            showTab(0);
        }
    }

    private void renderHistory() {
        LinearLayout body = scrollPage();
        heading(body, "History", "Encrypted entries kept only on this device");
        JSONArray items = readEntriesOrWarn(body);
        if (items == null) return;
        LinearLayout count = panel(body, true);
        count.addView(text(items.length() + " local entries", 19, WHITE, true));
        space(count, 5);
        count.addView(text("This is a journal, not a record of AI-generated dialogue.", 13, MUTED, false));
        if (items.length() > 0) {
            addAction(body, "Clear conversation history", () -> new AlertDialog.Builder(this)
                    .setTitle("Clear local conversation history?")
                    .setMessage("This will delete all journal entries from the encrypted store. Confirmed principles in Memory will remain untouched.")
                    .setNegativeButton("Keep history", null)
                    .setPositiveButton("Delete history", (d, w) -> {
                        try {
                            journal.clear();
                            notice = "Conversation history cleared.";
                            showTab(1);
                        } catch (Exception error) {
                            Toast.makeText(this, "Could not clear encrypted history", Toast.LENGTH_LONG).show();
                        }
                    }).show(), false);
        }
        if (items.length() == 0) {
            LinearLayout empty = panel(body, false);
            empty.addView(text("No notes yet. Start with Chat.", 14, MUTED, false));
        } else {
            for (int i = items.length() - 1; i >= 0; i--) {
                JSONObject entry = items.optJSONObject(i);
                if (entry != null) addMessage(body, entry);
            }
        }
    }

    private void renderMemory() {
        LinearLayout body = scrollPage();
        heading(body, "Memory", "Only principles you explicitly confirmed");
        LinearLayout guard = panel(body, true);
        guard.addView(text("CONFIRM BEFORE REMEMBERING", 11, ACCENT, true));
        space(guard, 8);
        guard.addView(text("Journal notes never become Forge memories automatically. Manual teaching still requires your explicit confirmation.", 14, WHITE, false));
        addAction(body, "Teach Forge a principle", () -> {
            Intent intent = new Intent(this, MainActivity.class);
            intent.putExtra("forge_manual_teach", true);
            startActivity(intent);
        }, true);
        try {
            JSONArray principles = new EncryptedMemory(this).principles();
            if (principles.length() == 0) {
                LinearLayout empty = panel(body, false);
                empty.addView(text("No principles confirmed yet.", 15, MUTED, false));
            }
            for (int i = principles.length() - 1; i >= 0; i--) {
                JSONObject item = principles.optJSONObject(i);
                if (item == null) continue;
                LinearLayout card = panel(body, false);
                card.addView(text(item.optString("principle"), 16, WHITE, true));
                space(card, 8);
                card.addView(text("WHY  " + item.optString("why"), 13, MUTED, false));
                space(card, 5);
                card.addView(text("SOURCE  " + item.optString("app"), 11, ACCENT, false));
                space(card, 5);
                card.addView(text(formatTime(item.optLong("confirmed_at")), 11, MUTED, false));
            }
            if (principles.length() > 0) {
                addAction(body, "Delete all confirmed principles", () ->
                        new AlertDialog.Builder(this)
                                .setTitle("Delete every confirmed principle?")
                                .setMessage("The encrypted principle vault will be cleared. Your conversation journal will remain untouched.")
                                .setNegativeButton("Cancel", null)
                                .setPositiveButton("Delete principles", (d, w) -> {
                                    try {
                                        new EncryptedMemory(this).forgetPrinciples();
                                        showTab(2);
                                    } catch (Exception error) {
                                        Toast.makeText(this, "Unable to delete principles", Toast.LENGTH_LONG).show();
                                    }
                                }).show(), false);
            }
        } catch (Exception error) {
            LinearLayout warning = panel(body, true);
            warning.addView(text("Confirmed memory vault could not be read. No data was deleted.", 14, WHITE, false));
        }
    }

    private void renderSettings() {
        LinearLayout body = scrollPage();
        heading(body, "Settings", "Privacy controls and intentional learning");
        LinearLayout privacy = panel(body, true);
        privacy.addView(text("OFFLINE BY DESIGN", 11, ACCENT, true));
        space(privacy, 8);
        privacy.addView(text("This build has no Internet permission, screen capture or content-reading accessibility service. Chat notes and confirmed principles remain in separate encrypted device stores.", 14, WHITE, false));
        space(body, 7);
        addAction(body, "Observation & permission controls", () ->
                startActivity(new Intent(this, MainActivity.class)), true);
        addAction(body, "Stop observation now", () -> {
            stopService(new Intent(this, ObserveService.class));
            Toast.makeText(this, "Stop requested for observation service", Toast.LENGTH_SHORT).show();
        }, false);
        LinearLayout explainer = panel(body, false);
        explainer.addView(text("Observation is optional", 16, WHITE, true));
        space(explainer, 8);
        explainer.addView(text("Only explicitly selected app-foreground changes can prompt a why-question. The foreground notification has a Stop control. Usage Access may require Android's restricted-settings approval after sideloading.", 13, MUTED, false));
        addAction(body, "Manual teaching without Usage Access", () -> {
            Intent intent = new Intent(this, MainActivity.class);
            intent.putExtra("forge_manual_teach", true);
            startActivity(intent);
        }, false);
    }

    private void renderDiagnostics() {
        LinearLayout body = scrollPage();
        heading(body, "Diagnostics", "Local runtime facts, not simulated AI abilities");
        ForgeModelAdapter.ModelStatus status = model.status();
        LinearLayout card = panel(body, true);
        card.addView(text("MODEL ADAPTER", 11, ACCENT, true));
        space(card, 8);
        card.addView(text(status.name, 18, WHITE, true));
        space(card, 5);
        card.addView(text("Available: " + (status.available ? "yes" : "no"), 13, MUTED, false));
        card.addView(text("Version: " + status.version, 13, MUTED, false));
        card.addView(text("Weights: " + status.provenance, 13, MUTED, false));
        space(card, 6);
        card.addView(text(status.detail, 13, WHITE, false));

        LinearLayout platform = panel(body, false);
        platform.addView(text("APP & DEVICE", 11, ACCENT, true));
        space(platform, 7);
        platform.addView(text("Package: " + getPackageName(), 13, WHITE, false));
        platform.addView(text("Android API: " + android.os.Build.VERSION.SDK_INT, 13, WHITE, false));
        platform.addView(text("Version: " + appVersion(), 13, WHITE, false));
        platform.addView(text("Network permission: " +
                (getPackageManager().checkPermission("android.permission.INTERNET", getPackageName())
                        == PackageManager.PERMISSION_GRANTED ? "granted" : "not requested"), 13, WHITE, false));
        platform.addView(text("Usage Access: " + (hasUsageAccess() ? "granted" : "not granted"),
                13, WHITE, false));

        LinearLayout storage = panel(body, false);
        storage.addView(text("LOCAL STORAGE", 11, ACCENT, true));
        space(storage, 7);
        try {
            storage.addView(text("Encrypted journal entries: " + journal.entries().length(),
                    13, WHITE, false));
        } catch (Exception error) {
            storage.addView(text("Encrypted journal: unavailable (data retained)",
                    13, WHITE, false));
        }
        try {
            EncryptedMemory vault = new EncryptedMemory(this);
            storage.addView(text("Confirmed principles: " + vault.principles().length(),
                    13, WHITE, false));
            storage.addView(text("Allowed observed apps: " + vault.allowed().size(),
                    13, WHITE, false));
        } catch (Exception error) {
            storage.addView(text("Principle vault: unavailable (data retained)",
                    13, WHITE, false));
        }
        space(storage, 8);
        storage.addView(text("At-rest protection: Android Keystore AES-GCM / app-private files. Automatic backups disabled.", 13, MUTED, false));

        LinearLayout limitations = panel(body, false);
        limitations.addView(text("NOT YET AVAILABLE", 11, ACCENT, true));
        space(limitations, 8);
        limitations.addView(text("Forge-native model inference, voice conversation, reasoning, autonomous actions, screen reading and independent on-device learning.", 13, WHITE, false));
    }

    private String appVersion() {
        try {
            return getPackageManager().getPackageInfo(getPackageName(), 0).versionName;
        } catch (Exception error) {
            return "unknown";
        }
    }

    private boolean hasUsageAccess() {
        AppOpsManager ops = (AppOpsManager) getSystemService(Context.APP_OPS_SERVICE);
        return ops != null && ops.checkOpNoThrow(AppOpsManager.OPSTR_GET_USAGE_STATS,
                Process.myUid(), getPackageName()) == AppOpsManager.MODE_ALLOWED;
    }
}
