package ai.forgecore.mobile;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.AppOpsManager;
import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Build;
import android.os.Bundle;
import android.os.Process;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.speech.tts.Voice;
import android.text.InputFilter;
import android.text.InputType;
import android.view.Gravity;
import android.view.Menu;
import android.view.View;
import android.view.WindowManager;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.PopupMenu;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.text.DateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.Set;

/**
 * Offline-first Forge communication shell with an optional, clearly labeled
 * pretrained local language bridge. The helper is not Forge Seed.
 * Existing v0.2 teaching, observation and memory vault remain in MainActivity.
 */
public final class ForgeHomeActivity extends Activity {
    private static final int REQUEST_GGUF = 4107;
    private static final int REQUEST_MICROPHONE = 4108;
    private static final int BG = Color.rgb(8, 8, 11);
    private static final int SURFACE = Color.rgb(17, 17, 22);
    private static final int SURFACE_RAISED = Color.rgb(25, 24, 31);
    private static final int ACCENT = Color.rgb(178, 105, 255);
    private static final int EMBER = Color.rgb(255, 139, 51);
    private static final int HOLO_CYAN = Color.rgb(98, 230, 255);
    private static final int HOLO_VIOLET = Color.rgb(183, 112, 255);
    private static final int WHITE = Color.rgb(244, 241, 247);
    private static final int MUTED = Color.rgb(157, 151, 167);
    private static final String[] TABS = {"Chat", "Chats & Projects", "Memory", "Settings", "Diagnostics"};

    private LocalLanguageBridge model;
    private EncryptedJournal journal;
    private FrameLayout content;
    private int selectedTab = 0;
    private boolean generating = false;
    private boolean responseStarted = false;
    private boolean listening = false;
    private boolean speaking = false;
    private boolean voiceTurn = false;
    private SpeechRecognizer speechRecognizer;
    private TextToSpeech textToSpeech;
    private boolean ttsReady = false;
    private String pendingSpeech;
    private ForgeEmberView avatarView;
    private TextView stateLabel;
    private TextView streamingView;
    private FrameLayout microphoneButton;
    private EditText inlineComposer;
    private StringBuilder streaming = new StringBuilder();
    private String notice = "";

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        model = new LocalLanguageBridge(this);
        journal = new EncryptedJournal(this);
        getWindow().setStatusBarColor(BG);
        getWindow().setNavigationBarColor(BG);
        getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE);

        LinearLayout root = column();
        root.setBackgroundColor(BG);
        root.setFitsSystemWindows(true);

        LinearLayout header = new LinearLayout(this);
        header.setGravity(Gravity.CENTER_VERTICAL);
        header.setPadding(dp(15), dp(8), dp(15), dp(8));
        header.setBackgroundColor(BG);
        TextView menuButton = text("☰", 23, MUTED, false);
        menuButton.setGravity(Gravity.CENTER);
        menuButton.setContentDescription("Open Forge sections");
        menuButton.setBackground(shape(SURFACE, 14));
        header.addView(menuButton, new LinearLayout.LayoutParams(dp(48), dp(48)));
        menuButton.setOnClickListener(view -> showNavigationMenu(menuButton));
        LinearLayout titleBlock = column();
        titleBlock.setGravity(Gravity.CENTER);
        TextView wordmark = text("✦  Forge", 22, WHITE, true);
        wordmark.setGravity(Gravity.CENTER);
        titleBlock.addView(wordmark);
        TextView strap = text("PRIVATE  ·  ON DEVICE", 9, MUTED, true);
        strap.setLetterSpacing(0.13f);
        strap.setGravity(Gravity.CENTER);
        titleBlock.addView(strap);
        header.addView(titleBlock, new LinearLayout.LayoutParams(0, -2, 1f));
        TextView headerSpacer = text("", 12, MUTED, false);
        header.addView(headerSpacer, new LinearLayout.LayoutParams(dp(48), dp(48)));
        root.addView(header);

        content = new FrameLayout(this);
        root.addView(content, new LinearLayout.LayoutParams(-1, 0, 1f));

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

    @Override protected void onPause() {
        if (listening) {
            stopVoiceRecognizer();
            notice = "Voice input stopped when Forge went to the background.";
        }
        super.onPause();
    }

    @Override protected void onDestroy() {
        stopVoiceRecognizer();
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
            textToSpeech = null;
        }
        super.onDestroy();
    }

    @Override public void onRequestPermissionsResult(int requestCode, String[] permissions,
                                                       int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode != REQUEST_MICROPHONE) return;
        if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            beginOnDeviceListening();
        } else {
            notice = "Microphone access was not granted. You can still use text chat.";
            showTab(0);
        }
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_GGUF && resultCode == RESULT_OK && data != null) {
            Uri uri = data.getData();
            if (uri != null) model.importModel(uri, operationListener("Model import"));
        }
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
        card.setBackground(new ForgeHoloDrawable(getResources().getDisplayMetrics().density,
                raised ? HOLO_VIOLET : HOLO_CYAN, raised ? HOLO_CYAN : HOLO_VIOLET));
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
        button.setBackground(prominent ? shape(ACCENT, 13) : new ForgeHoloDrawable(getResources().getDisplayMetrics().density, HOLO_CYAN, HOLO_VIOLET));
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
        avatarView = null;
        stateLabel = null;
        streamingView = null;
        microphoneButton = null;
        inlineComposer = null;
        switch (selectedTab) {
            case 0: renderChat(); break;
            case 1: renderHistory(); break;
            case 2: renderMemory(); break;
            case 3: renderSettings(); break;
            default: renderDiagnostics(); break;
        }
    }

    private void showNavigationMenu(View anchor) {
        PopupMenu popup = new PopupMenu(this, anchor);
        for (int i = 0; i < TABS.length; i++) popup.getMenu().add(Menu.NONE, i, i, TABS[i]);
        popup.setOnMenuItemClickListener(item -> {
            showTab(item.getItemId());
            return true;
        });
        popup.show();
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

        ForgeModelAdapter.ModelStatus modelStatus = model.status();

        LinearLayout conversationBar = new LinearLayout(this);
        conversationBar.setGravity(Gravity.CENTER_VERTICAL);
        conversationBar.setPadding(dp(20), dp(4), dp(20), dp(4));
        TextView currentConversation = text("◷  " + activeConversationTitle() + "  ▾", 12, MUTED, true);
        currentConversation.setSingleLine(true);
        currentConversation.setEllipsize(android.text.TextUtils.TruncateAt.END);
        currentConversation.setPadding(dp(12), dp(9), dp(12), dp(9));
        currentConversation.setBackground(new ForgeHoloDrawable(getResources().getDisplayMetrics().density, HOLO_VIOLET, HOLO_CYAN));
        currentConversation.setOnClickListener(v -> showConversationPicker());
        conversationBar.addView(currentConversation, new LinearLayout.LayoutParams(0, -2, 1f));
        TextView newConversation = text("＋", 22, ACCENT, true);
        newConversation.setGravity(Gravity.CENTER);
        newConversation.setContentDescription("Start a new conversation");
        newConversation.setOnClickListener(v -> createNewConversation());
        conversationBar.addView(newConversation, new LinearLayout.LayoutParams(dp(44), dp(42)));
        screen.addView(conversationBar);

        // Keep Forge's presence visible while the conversation itself scrolls.
        LinearLayout presence = column();
        presence.setPadding(dp(20), dp(2), dp(20), 0);
        screen.addView(presence, new LinearLayout.LayoutParams(-1, -2));
        ForgeEmberView avatar = new ForgeEmberView(this);
        avatarView = avatar;
        updateAvatarMode();
        LinearLayout.LayoutParams avatarParams = new LinearLayout.LayoutParams(-1, dp(238));
        avatarParams.bottomMargin = dp(3);
        presence.addView(avatar, avatarParams);

        TextView state = text("", 10, MUTED, true);
        state.setGravity(Gravity.CENTER);
        state.setLetterSpacing(0.13f);
        stateLabel = state;
        updateAvatarMode();
        presence.addView(state);

        ScrollView scrolling = new ScrollView(this);
        scrolling.setFillViewport(true);
        scrolling.setClipToPadding(false);
        LinearLayout.LayoutParams scrollingParams = new LinearLayout.LayoutParams(-1, 0, 1f);
        screen.addView(scrolling, scrollingParams);
        LinearLayout messages = column();
        messages.setPadding(dp(20), dp(8), dp(20), dp(14));
        scrolling.addView(messages);

        if (!notice.isEmpty()) {
            LinearLayout info = panel(messages, true);
            info.addView(text(notice, 13, ACCENT, false));
        }

        // A live response is newer than the submitted prompt, so put it first.
        if (generating) {
            LinearLayout live = panel(messages, true);
            live.addView(text(responseStarted ? "Local language bridge is responding…"
                    : "Local language bridge is thinking…", 13, ACCENT, true));
            if (streaming.length() > 0) {
                space(live, 8);
                streamingView = text(streaming.toString(), 14, WHITE, false);
                live.addView(streamingView);
            }
            addAction(live, "Cancel generation", () -> {
                model.cancel();
                generating = false;
                streaming.setLength(0);
                notice = "Generation canceled. No unfinished reply was saved.";
                showTab(0);
            }, false);
        }

        JSONArray entries = readEntriesOrWarn(messages);
        if (entries != null && entries.length() == 0) {
            TextView welcome = text("I’m here. What are you thinking about?", 19, WHITE, false);
            welcome.setGravity(Gravity.CENTER);
            welcome.setPadding(dp(12), dp(3), dp(12), dp(13));
            messages.addView(welcome);
            TextView explain = text(modelStatus.available
                    ? "Your reply will come from the optional local helper."
                    : "Type a private note, or set up the local language bridge in Settings to chat.",
                    12, MUTED, false);
            explain.setGravity(Gravity.CENTER);
            explain.setPadding(dp(18), 0, dp(18), dp(12));
            messages.addView(explain);
        } else if (entries != null) {
            // Journal stays chronological on disk; the chat view presents newest first.
            for (int i = entries.length() - 1; i >= 0; i--) {
                JSONObject entry = entries.optJSONObject(i);
                if (entry != null) addMessage(messages, entry);
            }
        }

        // The newest exchange is at the top; preserve the default top scroll position.
        LinearLayout composer = column();
        composer.setPadding(dp(16), dp(6), dp(16), dp(7));
        composer.setBackgroundColor(BG);
        FrameLayout inputWindow = new FrameLayout(this);
        inputWindow.setBackground(new ForgeHoloDrawable(getResources().getDisplayMetrics().density,
                HOLO_CYAN, HOLO_VIOLET));
        EditText input = new EditText(this);
        inlineComposer = input;
        input.setHint(modelStatus.available ? "Message the local language bridge…" : "Write a private note…");
        input.setHintTextColor(MUTED);
        input.setTextColor(WHITE);
        input.setTextSize(15);
        input.setGravity(Gravity.CENTER_VERTICAL);
        input.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
                | InputType.TYPE_TEXT_FLAG_MULTI_LINE);
        input.setSingleLine(false);
        input.setMinLines(1);
        input.setMaxLines(3);
        input.setFilters(new InputFilter[]{new InputFilter.LengthFilter(EncryptedJournal.MAX_MESSAGE_CHARS)});
        input.setImeOptions(EditorInfo.IME_ACTION_SEND | EditorInfo.IME_FLAG_NO_EXTRACT_UI);
        input.setPadding(dp(14), dp(8), dp(2), dp(8));
        input.setBackgroundColor(Color.TRANSPARENT);
        FrameLayout.LayoutParams inputParams = new FrameLayout.LayoutParams(-1, -2, Gravity.TOP | Gravity.LEFT);
        inputParams.leftMargin = dp(2);
        inputParams.rightMargin = dp(49);
        inputParams.topMargin = dp(2);
        inputWindow.addView(input, inputParams);
        inputWindow.setMinimumHeight(dp(56));
        TextView send = text("➤", 20, HOLO_CYAN, true);
        send.setGravity(Gravity.CENTER);
        send.setContentDescription(modelStatus.available ? "Send message" : "Save private note");
        send.setOnClickListener(view -> sendComposedPrompt());
        FrameLayout.LayoutParams sendParams = new FrameLayout.LayoutParams(dp(46), -1,
                Gravity.RIGHT | Gravity.CENTER_VERTICAL);
        sendParams.rightMargin = dp(3);
        inputWindow.addView(send, sendParams);
        LinearLayout.LayoutParams inputWindowParams = new LinearLayout.LayoutParams(-1, -2);
        inputWindowParams.bottomMargin = dp(5);
        composer.addView(inputWindow, inputWindowParams);
        input.setEnabled(!generating);
        send.setEnabled(!generating);
        input.setAlpha(generating ? .58f : 1f);
        send.setAlpha(generating ? .58f : 1f);
        input.setOnEditorActionListener((view, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_SEND) { sendComposedPrompt(); return true; }
            return false;
        });

        LinearLayout controls = new LinearLayout(this);
        controls.setGravity(Gravity.CENTER);
        FrameLayout microphone = new FrameLayout(this);
        microphoneButton = microphone;
        microphone.setContentDescription("Start on-device voice chat");
        microphone.setClipChildren(false);
        microphone.setClipToPadding(false);
        ForgeMicAuraView aura = new ForgeMicAuraView(this);
        aura.setActive(listening || speaking || generating);
        microphone.addView(aura, new FrameLayout.LayoutParams(-1, -1));
        FrameLayout micCore = new FrameLayout(this);
        micCore.setBackground(shape(listening ? EMBER : ACCENT, 40));
        micCore.setElevation(dp(5));
        micCore.addView(new ForgeMicIconView(this), new FrameLayout.LayoutParams(-1, -1));
        microphone.addView(micCore, new FrameLayout.LayoutParams(dp(66), dp(66), Gravity.CENTER));
        microphone.setOnClickListener(view -> toggleOnDeviceListening());
        controls.addView(microphone, new LinearLayout.LayoutParams(dp(148), dp(70)));

        View spacerRight = new View(this);
        controls.addView(spacerRight, new LinearLayout.LayoutParams(0, 1, 1f));
        TextView settings = text("⚙", 23, MUTED, false);
        settings.setGravity(Gravity.CENTER);
        settings.setContentDescription("Open settings");
        settings.setOnClickListener(view -> showTab(3));
        controls.addView(settings, new LinearLayout.LayoutParams(dp(48), dp(54)));
        composer.addView(controls, new LinearLayout.LayoutParams(-1, -2));
        screen.addView(composer);
    }

    private void sendComposedPrompt() {
        if (generating || inlineComposer == null) return;
        String prompt = inlineComposer.getText().toString().trim();
        if (prompt.isEmpty()) {
            Toast.makeText(this, "Write a message first", Toast.LENGTH_SHORT).show();
            return;
        }
        InputMethodManager keyboard = (InputMethodManager) getSystemService(INPUT_METHOD_SERVICE);
        if (keyboard != null) keyboard.hideSoftInputFromWindow(inlineComposer.getWindowToken(), 0);
        inlineComposer.setText("");
        submitPrompt(prompt);
    }

    private void addMessage(LinearLayout parent, JSONObject item) {
        boolean fromForge = "forge".equals(item.optString("role"));
        boolean fromTeacher = "teacher".equals(item.optString("role"));
        LinearLayout bubble = panel(parent, fromForge || fromTeacher);
        if (!fromForge && !fromTeacher) bubble.setBackground(new ForgeHoloDrawable(
                getResources().getDisplayMetrics().density, HOLO_CYAN, HOLO_VIOLET));
        String label = fromForge ? "FORGE SEED · NATIVE MODEL"
                : fromTeacher ? "LOCAL LANGUAGE BRIDGE · NOT FORGE SEED"
                : "YOU · PRIVATE MESSAGE";
        bubble.addView(text(label, 10, fromForge || fromTeacher ? HOLO_VIOLET : HOLO_CYAN, true));
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
        submitPrompt(prompt, false);
    }

    private void submitPrompt(String prompt, boolean fromVoice) {
        if (generating) return;
        voiceTurn = fromVoice;
        try {
            journal.append("user", prompt);
            notice = model.status().available
                    ? "Sent to the local language bridge. This helper is separate from Forge Seed."
                    : "Saved on this device. No model reply was generated.";
        } catch (Exception error) {
            notice = "Could not encrypt and save this message. No data was sent.";
            showTab(0);
            return;
        }

        if (!model.status().available) {
            voiceTurn = false;
            showTab(0);
            return;
        }

        generating = true;
        responseStarted = false;
        streaming = new StringBuilder();
        showTab(0);
        try {
            model.generate(prompt, new ForgeModelAdapter.Listener() {
                @Override public void onToken(String token) {
                    runOnUiThread(() -> {
                        if (!generating) return;
                        if (token != null && !token.isEmpty()) {
                            streaming.append(token);
                            if (!responseStarted) {
                                responseStarted = true;
                                updateAvatarMode();
                                if (streamingView == null && selectedTab == 0) showTab(0);
                            }
                            if (streamingView != null) streamingView.setText(streaming.toString());
                        }
                    });
                }
                @Override public void onComplete() {
                    runOnUiThread(() -> {
                        if (!generating) return;
                        generating = false;
                        responseStarted = false;
                        String answer = streaming.toString();
                        streaming.setLength(0);
                        boolean speakAnswer = voiceTurn;
                        voiceTurn = false;
                        if (!answer.trim().isEmpty()) {
                            try {
                                journal.append("teacher", answer);
                                notice = "Local language bridge response saved on this device.";
                            } catch (Exception error) {
                                notice = "Model replied, but secure storage failed; response was not saved.";
                            }
                        } else notice = "Local model finished without a reply.";
                        if (selectedTab == 0) showTab(0);
                        if (speakAnswer && !answer.trim().isEmpty()) speakReplyOffline(answer);
                    });
                }
                @Override public void onError(String reason) {
                    runOnUiThread(() -> {
                        if (!generating) return;
                        generating = false;
                        responseStarted = false;
                        voiceTurn = false;
                        streaming.setLength(0);
                        notice = "Local inference unavailable: " + (reason == null ? "unknown error" : reason);
                        if (selectedTab == 0) showTab(0);
                    });
                }
            });
        } catch (Exception error) {
            generating = false;
            responseStarted = false;
            voiceTurn = false;
            streaming.setLength(0);
            notice = "Local inference failed. Your note remains saved.";
            showTab(0);
        }
    }

    private void updateAvatarMode() {
        if (avatarView == null) return;
        ForgeEmberView.Mode mode = listening ? ForgeEmberView.Mode.LISTENING
                : speaking ? ForgeEmberView.Mode.SPEAKING
                : generating ? responseStarted ? ForgeEmberView.Mode.RESPONDING
                : ForgeEmberView.Mode.THINKING : ForgeEmberView.Mode.IDLE;
        avatarView.setMode(mode);
        if (stateLabel != null) {
            String label = listening ? "LISTENING ON DEVICE"
                    : speaking ? "SPEAKING"
                    : generating ? responseStarted ? "RESPONDING" : "THINKING"
                    : model.status().available ? "LOCAL LANGUAGE BRIDGE · READY" : "HERE WITH YOU";
            stateLabel.setText(label);
            stateLabel.setTextColor(listening || speaking || generating ? EMBER : MUTED);
        }
        if (microphoneButton != null) {
            View micCore = microphoneButton.getChildAt(1);
            if (micCore != null) micCore.setBackground(shape(listening ? EMBER : ACCENT, 40));
            if (microphoneButton.getChildAt(0) instanceof ForgeMicAuraView) {
                ((ForgeMicAuraView) microphoneButton.getChildAt(0)).setActive(listening || speaking || generating);
            }
            microphoneButton.setContentDescription(listening
                    ? "Stop listening" : "Start on-device voice chat");
        }
    }

    private void toggleOnDeviceListening() {
        if (listening) {
            stopVoiceRecognizer();
            notice = "Voice input stopped.";
            updateAvatarMode();
            return;
        }
        if (generating || speaking) {
            if (speaking && textToSpeech != null) {
                textToSpeech.stop();
                speaking = false;
                updateAvatarMode();
            } else {
                Toast.makeText(this, "Wait for the local reply to finish.", Toast.LENGTH_SHORT).show();
                return;
            }
        }
        if (Build.VERSION.SDK_INT < 31) {
            Toast.makeText(this,
                    "On-device voice input requires Android 12 or newer. Text chat is still available.",
                    Toast.LENGTH_LONG).show();
            return;
        }
        if (!SpeechRecognizer.isOnDeviceRecognitionAvailable(this)) {
            Toast.makeText(this,
                    "This phone has no on-device speech recognizer available. Forge will not send audio to an online recognizer.",
                    Toast.LENGTH_LONG).show();
            return;
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_MICROPHONE);
            return;
        }
        beginOnDeviceListening();
    }

    private void beginOnDeviceListening() {
        if (Build.VERSION.SDK_INT < 31 ||
                !SpeechRecognizer.isOnDeviceRecognitionAvailable(this)) return;
        stopVoiceRecognizer();
        try {
            speechRecognizer = SpeechRecognizer.createOnDeviceSpeechRecognizer(this);
            speechRecognizer.setRecognitionListener(new RecognitionListener() {
                @Override public void onReadyForSpeech(Bundle params) {
                    listening = true;
                    updateAvatarMode();
                }
                @Override public void onBeginningOfSpeech() {
                }
                @Override public void onRmsChanged(float rmsdB) {
                    if (avatarView != null) avatarView.setAudioLevel(rmsdB);
                }
                @Override public void onBufferReceived(byte[] buffer) {}
                @Override public void onEndOfSpeech() {
                    if (stateLabel != null) stateLabel.setText("PROCESSING ON DEVICE");
                }
                @Override public void onError(int error) {
                    releaseVoiceRecognizer();
                    if (error != SpeechRecognizer.ERROR_CLIENT) {
                        notice = error == SpeechRecognizer.ERROR_NO_MATCH ||
                                error == SpeechRecognizer.ERROR_SPEECH_TIMEOUT
                                ? "I didn’t catch that. Tap the mic and try again."
                                : "On-device voice input stopped. You can retry or use text chat.";
                        showTab(0);
                    }
                }
                @Override public void onResults(Bundle results) {
                    String phrase = firstRecognition(results);
                    releaseVoiceRecognizer();
                    if (phrase.isEmpty()) {
                        notice = "I didn’t catch that. Tap the mic and try again.";
                        showTab(0);
                    } else {
                        notice = "Voice message recognized on this device.";
                        submitPrompt(phrase, true);
                    }
                }
                @Override public void onPartialResults(Bundle partialResults) {
                    String phrase = firstRecognition(partialResults);
                    if (!phrase.isEmpty() && stateLabel != null) stateLabel.setText("HEARING · " + phrase);
                }
                @Override public void onEvent(int eventType, Bundle params) {}
            });
            Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
            intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                    RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
            intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault().toLanguageTag());
            intent.putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true);
            intent.putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1);
            listening = true;
            notice = "Listening with Android’s on-device speech recognizer. Audio is not saved.";
            updateAvatarMode();
            speechRecognizer.startListening(intent);
        } catch (Exception error) {
            releaseVoiceRecognizer();
            notice = "Could not start on-device voice input. Text chat is still available.";
            showTab(0);
        }
    }

    private String firstRecognition(Bundle results) {
        if (results == null) return "";
        java.util.ArrayList<String> matches =
                results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
        return matches == null || matches.isEmpty() || matches.get(0) == null
                ? "" : matches.get(0).trim();
    }

    private void stopVoiceRecognizer() {
        if (speechRecognizer != null) {
            try { speechRecognizer.cancel(); } catch (Exception ignored) {}
            try { speechRecognizer.destroy(); } catch (Exception ignored) {}
            speechRecognizer = null;
        }
        listening = false;
    }

    private void releaseVoiceRecognizer() {
        if (speechRecognizer != null) {
            try { speechRecognizer.destroy(); } catch (Exception ignored) {}
            speechRecognizer = null;
        }
        listening = false;
        updateAvatarMode();
    }

    private void speakReplyOffline(String reply) {
        pendingSpeech = reply;
        if (textToSpeech != null) {
            if (ttsReady) startOfflineSpeech();
            return;
        }
        textToSpeech = new TextToSpeech(this, status -> runOnUiThread(() -> {
            if (status != TextToSpeech.SUCCESS || textToSpeech == null) {
                ttsReady = false;
                pendingSpeech = null;
                notice = "Reply saved. This phone has no available offline speech voice.";
                if (selectedTab == 0) showTab(0);
                return;
            }
            int language = textToSpeech.setLanguage(Locale.US);
            Set<Voice> voices = textToSpeech.getVoices();
            Voice offlineEnglish = null;
            if (voices != null) {
                for (Voice voice : voices) {
                    if (voice.getLocale() != null && "en".equals(voice.getLocale().getLanguage())
                            && !voice.isNetworkConnectionRequired()) {
                        offlineEnglish = voice;
                        break;
                    }
                }
            }
            ttsReady = language != TextToSpeech.LANG_MISSING_DATA &&
                    language != TextToSpeech.LANG_NOT_SUPPORTED && offlineEnglish != null;
            if (ttsReady) {
                textToSpeech.setVoice(offlineEnglish);
                textToSpeech.setSpeechRate(.96f);
                textToSpeech.setOnUtteranceProgressListener(new UtteranceProgressListener() {
                    @Override public void onStart(String utteranceId) {
                        runOnUiThread(() -> {
                            speaking = true;
                            updateAvatarMode();
                        });
                    }
                    @Override public void onDone(String utteranceId) {
                        runOnUiThread(() -> {
                            speaking = false;
                            updateAvatarMode();
                        });
                    }
                    @Override public void onError(String utteranceId) {
                        runOnUiThread(() -> {
                            speaking = false;
                            updateAvatarMode();
                        });
                    }
                });
                startOfflineSpeech();
            } else {
                pendingSpeech = null;
                notice = "Reply saved. Install an offline English voice in Android Text-to-Speech settings to hear replies.";
                if (selectedTab == 0) showTab(0);
            }
        }));
    }

    private void startOfflineSpeech() {
        if (textToSpeech == null || !ttsReady || pendingSpeech == null) return;
        String reply = pendingSpeech;
        pendingSpeech = null;
        int result = textToSpeech.speak(reply, TextToSpeech.QUEUE_FLUSH, null, "forge-local-reply");
        if (result == TextToSpeech.ERROR) {
            speaking = false;
            notice = "Reply saved, but offline speech could not start.";
            updateAvatarMode();
        }
    }

    private void renderHistory() {
        LinearLayout body = scrollPage();
        heading(body, "Chats & Projects", "Keep separate conversations, then gather related chats into project spaces");
        LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);
        TextView newChat = action("＋  New conversation", this::createNewConversation, true);
        actions.addView(newChat, new LinearLayout.LayoutParams(0, -2, 1f));
        View gap = new View(this);
        actions.addView(gap, new LinearLayout.LayoutParams(dp(10), 1));
        TextView newProject = action("＋  New project", this::createProject, false);
        actions.addView(newProject, new LinearLayout.LayoutParams(0, -2, 1f));
        LinearLayout.LayoutParams actionsParams = new LinearLayout.LayoutParams(-1, -2);
        actionsParams.bottomMargin = dp(18);
        body.addView(actions, actionsParams);

        LinearLayout projectsSection = panel(body, true);
        projectsSection.addView(text("PROJECT SPACES", 11, ACCENT, true));
        space(projectsSection, 10);
        JSONArray projects;
        try { projects = journal.projects(); }
        catch (Exception error) { projects = null; }
        if (projects == null) {
            projectsSection.addView(text("Encrypted project list unavailable. Existing data has not been deleted.", 13, MUTED, false));
        } else if (projects.length() == 0) {
            projectsSection.addView(text("No project spaces yet. Create one to group conversations that belong to the same piece of work.", 13, MUTED, false));
        } else {
            for (int i = projects.length() - 1; i >= 0; i--) {
                JSONObject project = projects.optJSONObject(i);
                if (project == null) continue;
                String projectId = project.optString("id");
                TextView row = text("◈  " + project.optString("title") + "     ›", 15, WHITE, true);
                row.setPadding(dp(4), dp(10), dp(4), dp(10));
                row.setOnClickListener(v -> renderProject(projectId));
                projectsSection.addView(row);
            }
        }

        space(body, 5);
        body.addView(text("CONVERSATIONS", 11, MUTED, true));
        space(body, 9);
        JSONArray conversations;
        try { conversations = journal.conversations(); }
        catch (Exception error) {
            LinearLayout warning = panel(body, true);
            warning.addView(text("Encrypted conversation list unavailable. Existing data has not been deleted.", 13, MUTED, false));
            return;
        }
        LinearLayout count = panel(body, false);
        count.addView(text(conversations.length() + " conversations · " + safeEntryCount() + " encrypted messages", 13, MUTED, false));
        if (conversations.length() > 0) {
            addAction(body, "Clear conversation history", () -> new AlertDialog.Builder(this)
                    .setTitle("Clear all local conversation messages?")
                    .setMessage(model.status().available
                            ? "This deletes messages from every conversation and unloads the model. Conversation and project organization will remain. The imported model file and confirmed principles remain."
                            : "This deletes messages from every conversation. Conversation and project organization will remain; confirmed principles in Memory are untouched.")
                    .setNegativeButton("Keep history", null)
                    .setPositiveButton("Delete history", (d, w) -> {
                        if (generating) {
                            Toast.makeText(this, "Cancel the current reply before clearing history.", Toast.LENGTH_LONG).show();
                            return;
                        }
                        if (model.status().available) {
                            model.unloadModel(new ForgeModelAdapter.OperationListener() {
                                @Override public void onComplete(String message) {
                                    runOnUiThread(() -> clearConversationHistory());
                                }
                                @Override public void onError(String reason) {
                                    runOnUiThread(() -> {
                                        notice = "History was kept because the model session could not be cleared: " + reason;
                                        showTab(1);
                                        Toast.makeText(ForgeHomeActivity.this, notice, Toast.LENGTH_LONG).show();
                                    });
                                }
                            });
                        } else {
                            clearConversationHistory();
                        }
                    }).show(), false);
        }
        if (conversations.length() == 0) {
            LinearLayout empty = panel(body, false);
            empty.addView(text("No conversations yet. Start one above.", 14, MUTED, false));
        } else {
            String activeId = "";
            try { activeId = journal.activeConversationId(); } catch (Exception ignored) { }
            for (int i = conversations.length() - 1; i >= 0; i--) {
                JSONObject conversation = conversations.optJSONObject(i);
                if (conversation != null) addConversationCard(body, conversation, activeId);
            }
        }
    }

    private void addConversationCard(LinearLayout parent, JSONObject conversation, String activeId) {
        String id = conversation.optString("id");
        String title = conversation.optString("title", "Conversation");
        JSONArray entries = conversation.optJSONArray("entries");
        String preview = "No messages yet";
        long updated = conversation.optLong("updated_at");
        if (entries != null && entries.length() > 0) {
            JSONObject last = entries.optJSONObject(entries.length() - 1);
            if (last != null) preview = last.optString("text", "").replaceAll("\\s+", " ");
            if (last != null) updated = last.optLong("created_at", updated);
            if (preview.length() > 90) preview = preview.substring(0, 87) + "…";
        }
        LinearLayout card = panel(parent, id.equals(activeId));
        TextView titleView = text((id.equals(activeId) ? "●  " : "") + title, 16, WHITE, true);
        titleView.setOnClickListener(v -> openConversation(id));
        card.addView(titleView);
        space(card, 5);
        TextView detail = text(preview + "\n" + formatTime(updated), 12, MUTED, false);
        detail.setOnClickListener(v -> openConversation(id));
        card.addView(detail);
        space(card, 7);
        TextView organize = text("Add to project  ·  Open conversation", 12, ACCENT, true);
        organize.setOnClickListener(v -> new AlertDialog.Builder(this)
                .setTitle("Organize conversation")
                .setItems(new String[]{"Open conversation", "Add to project"}, (dialog, which) -> {
                    if (which == 0) openConversation(id); else chooseProjectForConversation(id);
                }).show());
        card.addView(organize);
    }

    private void renderProject(String projectId) {
        try {
            JSONObject project = findById(journal.projects(), projectId);
            if (project == null) { showTab(1); return; }
            LinearLayout body = scrollPage();
            heading(body, project.optString("title"), "Project space · related conversations stay linked here");
            addAction(body, "＋  Add current conversation", () -> {
                try {
                    journal.linkActiveConversationToProject(projectId);
                    notice = "Conversation added to " + project.optString("title") + ".";
                    renderProject(projectId);
                } catch (Exception error) {
                    Toast.makeText(this, "Could not link this conversation", Toast.LENGTH_LONG).show();
                }
            }, true);
            space(body, 8);
            body.addView(text("LINKED CONVERSATIONS", 11, MUTED, true));
            space(body, 8);
            JSONArray conversations = journal.conversations();
            int linked = 0;
            for (int i = conversations.length() - 1; i >= 0; i--) {
                JSONObject conversation = conversations.optJSONObject(i);
                if (conversation == null || !contains(conversation.optJSONArray("project_ids"), projectId)) continue;
                linked++;
                addConversationCard(body, conversation, journal.activeConversationId());
            }
            if (linked == 0) {
                LinearLayout empty = panel(body, false);
                empty.addView(text("This project has no conversations yet. Add the current chat, or open Chats & Projects to link another conversation.", 13, MUTED, false));
            }
            addAction(body, "Back to Chats & Projects", () -> showTab(1), false);
        } catch (Exception error) {
            Toast.makeText(this, "Could not open this project space", Toast.LENGTH_LONG).show();
            showTab(1);
        }
    }

    private JSONObject findById(JSONArray items, String id) {
        if (items == null) return null;
        for (int i = 0; i < items.length(); i++) {
            JSONObject item = items.optJSONObject(i);
            if (item != null && id.equals(item.optString("id"))) return item;
        }
        return null;
    }

    private boolean contains(JSONArray items, String value) {
        if (items == null) return false;
        for (int i = 0; i < items.length(); i++) if (value.equals(items.optString(i))) return true;
        return false;
    }

    private void chooseProjectForConversation(String conversationId) {
        try {
            JSONArray projects = journal.projects();
            if (projects.length() == 0) {
                new AlertDialog.Builder(this).setTitle("No project spaces yet")
                        .setMessage("Create a project space first, then link conversations to it.")
                        .setNegativeButton("Close", null).setPositiveButton("Create project", (d, w) -> createProject()).show();
                return;
            }
            String[] labels = new String[projects.length()];
            for (int i = 0; i < projects.length(); i++) labels[i] = projects.optJSONObject(i).optString("title");
            new AlertDialog.Builder(this).setTitle("Add conversation to project")
                    .setItems(labels, (d, which) -> {
                        try {
                            journal.linkConversationToProject(conversationId, projects.optJSONObject(which).optString("id"));
                            showTab(1);
                        } catch (Exception error) { Toast.makeText(this, "Could not link conversation", Toast.LENGTH_LONG).show(); }
                    }).show();
        } catch (Exception error) { Toast.makeText(this, "Could not load project spaces", Toast.LENGTH_LONG).show(); }
    }

    private String activeConversationTitle() {
        try {
            String id = journal.activeConversationId();
            JSONObject conversation = findById(journal.conversations(), id);
            return conversation == null ? "New conversation" : conversation.optString("title", "New conversation");
        } catch (Exception error) { return "Conversation"; }
    }

    private void showConversationPicker() {
        if (generating) {
            Toast.makeText(this, "Wait for the reply or cancel it before switching conversations", Toast.LENGTH_LONG).show();
            return;
        }
        try {
            JSONArray items = journal.conversations();
            String[] labels = new String[items.length()];
            for (int i = 0; i < items.length(); i++) labels[i] = items.optJSONObject(i).optString("title", "Conversation");
            new AlertDialog.Builder(this).setTitle("Switch conversation").setItems(labels, (d, which) -> {
                try { journal.activateConversation(items.optJSONObject(which).optString("id")); showTab(0); }
                catch (Exception error) { Toast.makeText(this, "Could not open conversation", Toast.LENGTH_LONG).show(); }
            }).setPositiveButton("Chats & Projects", (d, w) -> showTab(1)).setNegativeButton("Close", null).show();
        } catch (Exception error) { Toast.makeText(this, "Could not load conversations", Toast.LENGTH_LONG).show(); }
    }

    private void openConversation(String id) {
        if (generating) { Toast.makeText(this, "Wait for the reply or cancel it before switching conversations", Toast.LENGTH_LONG).show(); return; }
        try { journal.activateConversation(id); showTab(0); }
        catch (Exception error) { Toast.makeText(this, "Could not open conversation", Toast.LENGTH_LONG).show(); }
    }

    private void createNewConversation() {
        if (generating) { Toast.makeText(this, "Wait for the reply or cancel it before starting a new conversation", Toast.LENGTH_LONG).show(); return; }
        try { journal.createConversation(); notice = "New private conversation started."; showTab(0); }
        catch (Exception error) { Toast.makeText(this, "Could not create conversation", Toast.LENGTH_LONG).show(); }
    }

    private void createProject() {
        EditText input = new EditText(this);
        input.setHint("Project name");
        input.setTextColor(WHITE);
        input.setHintTextColor(MUTED);
        input.setSingleLine(true);
        input.setFilters(new InputFilter[]{new InputFilter.LengthFilter(60)});
        new AlertDialog.Builder(this).setTitle("Create project space").setView(input)
                .setNegativeButton("Cancel", null).setPositiveButton("Create", (d, w) -> {
                    try { journal.createProject(input.getText().toString()); showTab(1); }
                    catch (Exception error) { Toast.makeText(this, "Enter a project name (up to 60 characters)", Toast.LENGTH_LONG).show(); }
                }).show();
    }

    private int safeEntryCount() {
        try { return journal.entryCount(); } catch (Exception error) { return 0; }
    }

    private void clearConversationHistory() {
        try {
            journal.clear();
            notice = "Conversation history cleared. The local model is unloaded.";
            showTab(1);
        } catch (Exception error) {
            Toast.makeText(this, "Could not clear encrypted history", Toast.LENGTH_LONG).show();
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
        heading(body, "Settings", "Local chat model and privacy controls");
        renderLanguageBridgeControls(body);
        space(body, 5);
        LinearLayout privacy = panel(body, true);
        privacy.addView(text("OFFLINE BY DESIGN", 11, ACCENT, true));
        space(privacy, 8);
        privacy.addView(text("This build has no Internet permission, screen capture or content-reading accessibility service. Voice uses Android’s on-device recognizer only when the phone provides one, plus an installed offline speech voice. Audio is not saved; recognized text enters the encrypted journal only after you speak a message.", 14, WHITE, false));
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

    private void renderLanguageBridgeControls(LinearLayout body) {
        ForgeModelAdapter.ModelStatus status = model.status();
        LinearLayout bridge = panel(body, true);
        bridge.addView(text("LOCAL LANGUAGE BRIDGE", 11, ACCENT, true));
        space(bridge, 8);
        bridge.addView(text("Optional pretrained helper · separate from Forge Seed", 16, WHITE, true));
        space(bridge, 6);
        bridge.addView(text(status.name + "\n" + status.provenance + "\n" + status.detail,
                13, MUTED, false));
        if (model.isBusy()) {
            // Progress is shown above; model and file actions stay disabled until it finishes.
        } else if (!status.available) {
            space(bridge, 8);
            bridge.addView(text("Choose a GGUF model you have permission to use. The app copies it into private storage. Forge Core does not download or upload the model.",
                    13, WHITE, false));
            space(bridge, 10);
            if (!model.hasImportedModelFile()) {
                addAction(bridge, "Download test teacher · Qwen 0.5B · about 491 MB",
                        this::confirmTeacherDownload, false);
            }
            addAction(bridge, "Choose local GGUF model", () -> {
                Intent picker = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                picker.addCategory(Intent.CATEGORY_OPENABLE);
                picker.setType("*/*");
                startActivityForResult(picker, REQUEST_GGUF);
            }, true);
            addAction(bridge, "Load local model", () ->
                    model.loadModel(operationListener("Model load")), false);
            if (model.hasImportedModelFile()) {
                addAction(bridge, "Remove imported model file", () -> new AlertDialog.Builder(this)
                        .setTitle("Remove local model file?")
                        .setMessage("This removes the imported GGUF from Forge Core private storage. The encrypted conversation history and confirmed principles remain.")
                        .setNegativeButton("Keep model", null)
                        .setPositiveButton("Remove model", (dialog, which) ->
                                model.removeModelFile(operationListener("Model removal")))
                        .show(), false);
            }
        } else {
            space(bridge, 10);
            addAction(bridge, "Unload model from memory", () ->
                    model.unloadModel(operationListener("Model unload")), false);
        }
    }

    private void confirmTeacherDownload() {
        new AlertDialog.Builder(this)
                .setTitle("Download the temporary test teacher?")
                .setMessage("This optional Qwen 0.5B GGUF is about 491 MB and uses an Apache 2.0 license. Android will open the download in your browser. The app itself has no Internet permission. It is a separate pretrained helper; it will not become Forge Seed or enter Forge training.")
                .setNegativeButton("Cancel", null)
                .setPositiveButton("Open browser", (dialog, which) -> {
                    Uri modelUri = Uri.parse("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf?download=true");
                    try {
                        startActivity(new Intent(Intent.ACTION_VIEW, modelUri));
                    } catch (Exception error) {
                        Toast.makeText(this, "Could not open the model download in a browser.", Toast.LENGTH_LONG).show();
                    }
                }).show();
    }

    private ForgeModelAdapter.OperationListener operationListener(String label) {
        return new ForgeModelAdapter.OperationListener() {
            @Override public void onComplete(String message) {
                runOnUiThread(() -> {
                    notice = message;
                    if (content != null) showTab(selectedTab);
                    Toast.makeText(ForgeHomeActivity.this, label + " complete", Toast.LENGTH_SHORT).show();
                });
            }
            @Override public void onError(String reason) {
                runOnUiThread(() -> {
                    notice = reason;
                    if (content != null) showTab(selectedTab);
                    Toast.makeText(ForgeHomeActivity.this, reason, Toast.LENGTH_LONG).show();
                });
            }
            @Override public void onProgress(String message) {
                runOnUiThread(() -> {
                    notice = message;
                    if (content != null) showTab(selectedTab);
                });
            }
        };
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
            storage.addView(text("Encrypted journal messages: " + journal.entryCount(),
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
        limitations.addView(text("Forge Seed chat inference, autonomous actions, screen reading and independent on-device learning.", 13, WHITE, false));
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
