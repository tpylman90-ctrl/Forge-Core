package ai.forgecore.mobile;

import android.Manifest;
import android.app.AlertDialog;
import android.app.AppOpsManager;
import android.app.usage.UsageEvents;
import android.app.usage.UsageStatsManager;
import android.content.Context;
import android.content.Intent;
import android.content.ActivityNotFoundException;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Process;
import android.provider.Settings;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Set;

/** First standalone Android proof of learning: no network or screen reading. */
public final class MainActivity extends android.app.Activity {
    private TextView status;

    @Override public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setStatusBarColor(Color.rgb(14, 22, 36));
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(26, 28, 26, 30);
        root.setBackgroundColor(Color.rgb(14, 22, 36));
        ScrollView scroll = new ScrollView(this);
        scroll.addView(root);
        setContentView(scroll);

        label(root, "FORGE CORE", 28, Color.rgb(100, 206, 242));
        label(root, "First Breath · Android learning companion", 16, Color.WHITE);
        label(root, "You choose what I may observe. I ask why, but I only remember what you confirm.", 15, Color.LTGRAY);
        label(root, "No screen capture, messages, internet permission or automatic model training.", 13, Color.LTGRAY);

        status = label(root, "", 15, Color.rgb(100, 206, 242));
        label(root, "Android 16 may restrict Usage Access for apps installed from downloaded APKs. First open Forge's own app details, tap the three-dot menu, and select Allow restricted settings if offered. Then open Usage Access.", 14, Color.LTGRAY);
        button(root, "1 · Open Forge app details (restricted settings)", this::openForgeAppDetails);
        button(root, "2 · Open Usage Access settings", this::openUsageAccess);
        button(root, "3 · Choose apps Forge may learn from", this::selectApps);
        button(root, "4 · Start observation (visible notification)", this::startObserving);
        button(root, "Help: Usage Access not showing Forge", this::showAccessHelp);
        button(root, "Pause / stop observation", () -> {
            stopService(new Intent(this, ObserveService.class));
            toast("Observation stopped");
        });
        button(root, "Review confirmed memories", this::reviewMemories);
        button(root, "Teach Forge a decision manually", () -> showQuestion("Your own example"));
        label(root, "Manual learning works even without Usage Access. This is still a template-based learner; genuine conversational language inference is a separate upcoming module.", 13, Color.LTGRAY);
        label(root, "Questions appear as notifications after selected app switches. This is a test prototype. You can stop observation at any time.", 13, Color.LTGRAY);

        refreshStatus();
        handleQuestion(getIntent());
    }

    @Override protected void onResume() {
        super.onResume();
        if (status != null) refreshStatus();
    }

    @Override protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        handleQuestion(intent);
    }

    private TextView label(LinearLayout root, String content, int size, int color) {
        TextView view = new TextView(this);
        view.setText(content);
        view.setTextSize(size);
        view.setTextColor(color);
        view.setPadding(0, 10, 0, 16);
        root.addView(view);
        return view;
    }

    private void button(LinearLayout root, String title, Runnable action) {
        Button view = new Button(this);
        view.setText(title);
        view.setAllCaps(false);
        view.setOnClickListener(v -> action.run());
        root.addView(view, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
    }

    private void toast(String text) {
        Toast.makeText(this, text, Toast.LENGTH_LONG).show();
    }

    private boolean hasUsageAccess() {
        AppOpsManager ops = (AppOpsManager) getSystemService(Context.APP_OPS_SERVICE);
        return ops.checkOpNoThrow(AppOpsManager.OPSTR_GET_USAGE_STATS,
                Process.myUid(), getPackageName()) == AppOpsManager.MODE_ALLOWED;
    }

    private void refreshStatus() {
        String access = hasUsageAccess() ? "GRANTED" : "NOT GRANTED";
        try {
            status.setText("Usage access: " + access + "  ·  Allowed apps: "
                    + new EncryptedMemory(this).allowed().size());
        } catch (Exception error) {
            status.setText("Memory unavailable — do not start observation until reviewed");
        }
    }

    private void openForgeAppDetails() {
        Intent details = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                Uri.parse("package:" + getPackageName()));
        try {
            startActivity(details);
        } catch (ActivityNotFoundException error) {
            toast("Open Android Settings > Apps > Forge Core manually");
        }
    }

    private void openUsageAccess() {
        try {
            startActivity(new Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS));
        } catch (ActivityNotFoundException error) {
            new AlertDialog.Builder(this)
                    .setTitle("Usage Access settings unavailable")
                    .setMessage("Open Android Settings > Apps > Special app access > Usage access. Device settings vary by manufacturer.")
                    .setPositiveButton("Open Forge details", (d, w) -> openForgeAppDetails())
                    .setNegativeButton("Close", null)
                    .show();
        }
    }

    private void showAccessHelp() {
        new AlertDialog.Builder(this)
                .setTitle("Android 16 Usage Access troubleshooting")
                .setMessage("1. Open Forge Core app details using the first button.\n\n"
                        + "2. If the upper-right three-dot menu includes 'Allow restricted settings', select it. Android may ask you to authenticate.\n\n"
                        + "3. Return to Forge and open Usage Access settings. Look for Forge Core and grant usage access.\n\n"
                        + "4. If Forge still does not appear, go to Android Settings > Apps > Special app access > Usage access, or search Settings for Usage access.\n\n"
                        + "5. You can use 'Teach Forge a decision manually' without this permission.\n\n"
                        + "Not all Motorola/Android versions show the same options. Forge cannot override this system restriction.")
                .setPositiveButton("Forge app details", (d, w) -> openForgeAppDetails())
                .setNegativeButton("Close", null)
                .show();
    }

    private void selectApps() {
        if (!hasUsageAccess()) {
            showAccessHelp();
            return;
        }
        try {
            long now = System.currentTimeMillis();
            UsageStatsManager usage = (UsageStatsManager) getSystemService(USAGE_STATS_SERVICE);
            UsageEvents events = usage.queryEvents(now - 24L * 3600_000L, now);
            Set<String> seen = new LinkedHashSet<>();
            UsageEvents.Event event = new UsageEvents.Event();
            if (events != null) {
                while (events.hasNextEvent()) {
                    events.getNextEvent(event);
                    if (event.getEventType() != UsageEvents.Event.ACTIVITY_RESUMED) continue;
                    String pkg = event.getPackageName();
                    if (pkg != null && !pkg.equals(getPackageName()) && !pkg.equals("android")
                            && !pkg.startsWith("com.android.systemui")) seen.add(pkg);
                }
            }
            ArrayList<String> sorted = new ArrayList<>(seen);
            Collections.sort(sorted);
            if (sorted.isEmpty()) {
                toast("No recent app transitions found; use other apps and try again");
                return;
            }
            String[] names = sorted.toArray(new String[0]);
            Set<String> allowed = new EncryptedMemory(this).allowed();
            boolean[] checks = new boolean[names.length];
            for (int i = 0; i < names.length; i++) checks[i] = allowed.contains(names[i]);
            new AlertDialog.Builder(this)
                    .setTitle("Choose apps to observe")
                    .setMultiChoiceItems(names, checks, (dialog, which, selected) -> checks[which] = selected)
                    .setPositiveButton("Save selection", (dialog, which) -> {
                        try {
                            Set<String> selection = new LinkedHashSet<>();
                            for (int i = 0; i < names.length; i++) if (checks[i]) selection.add(names[i]);
                            new EncryptedMemory(this).setAllowed(selection);
                            toast("Selected " + selection.size() + " apps. Restart observer if running.");
                            refreshStatus();
                        } catch (Exception error) {
                            toast("Could not save selection: " + error.getClass().getSimpleName());
                        }
                    })
                    .setNegativeButton("Cancel", null)
                    .show();
        } catch (Exception error) {
            toast("Could not list recent app activity: " + error.getClass().getSimpleName());
        }
    }

    private void startObserving() {
        if (!hasUsageAccess()) {
            showAccessHelp();
            return;
        }
        try {
            if (new EncryptedMemory(this).allowed().isEmpty()) {
                toast("Select at least one app first");
                return;
            }
        } catch (Exception error) {
            toast("Memory store unavailable");
            return;
        }
        if (Build.VERSION.SDK_INT >= 33 &&
                checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 11);
            toast("Allow notifications, then tap Start again");
            return;
        }
        try {
            startForegroundService(new Intent(this, ObserveService.class));
            toast("Requested observation start — stop from Forge or its notification");
        } catch (Exception error) {
            toast("Could not start observer: " + error.getClass().getSimpleName());
        }
    }

    private void handleQuestion(Intent intent) {
        if (intent == null || !intent.hasExtra("forge_question_app")) return;
        String pkg = intent.getStringExtra("forge_question_app");
        intent.removeExtra("forge_question_app");
        if (pkg == null || pkg.trim().isEmpty()) return;
        try {
            if (!new EncryptedMemory(this).allowed().contains(pkg)) return;
        } catch (Exception error) {
            return;
        }
        showQuestion(pkg);
    }

    private void showQuestion(String app) {
        LinearLayout form = new LinearLayout(this);
        form.setOrientation(LinearLayout.VERTICAL);
        form.setPadding(28, 10, 28, 0);
        EditText reason = new EditText(this);
        reason.setSingleLine(false);
        reason.setMinLines(2);
        reason.setHint("What were you trying to accomplish?");
        form.addView(reason);
        EditText principle = new EditText(this);
        principle.setSingleLine(false);
        principle.setMinLines(2);
        principle.setHint("What general lesson should Forge learn?");
        form.addView(principle);
        new AlertDialog.Builder(this)
                .setTitle("Forge asks why · " + app)
                .setMessage("I'm curious about your reasoning. Nothing will be saved unless you confirm.")
                .setView(form)
                .setPositiveButton("Confirm & remember", (dialog, which) -> {
                    try {
                        new EncryptedMemory(this).remember(app,
                                reason.getText().toString(), principle.getText().toString());
                        toast("Confirmed principle saved on this device");
                    } catch (Exception error) {
                        toast("Not saved. Complete both answers and retry.");
                    }
                })
                .setNegativeButton("Skip", null)
                .show();
    }

    private void reviewMemories() {
        try {
            JSONArray items = new EncryptedMemory(this).principles();
            StringBuilder text = new StringBuilder();
            for (int i = 0; i < items.length(); i++) {
                JSONObject record = items.getJSONObject(i);
                text.append(i + 1).append(". ").append(record.optString("principle"))
                        .append("\n  Why: ").append(record.optString("why")).append("\n\n");
            }
            if (items.length() == 0) text.append("No principles confirmed yet.");
            new AlertDialog.Builder(this)
                    .setTitle("Confirmed local memories")
                    .setMessage(text.toString())
                    .setPositiveButton("Done", null)
                    .setNeutralButton("Delete all memories", (d, w) ->
                            new AlertDialog.Builder(this).setTitle("Delete every learned principle?")
                                    .setMessage("This removes the saved principles from Forge's encrypted local vault.")
                                    .setNegativeButton("Cancel", null)
                                    .setPositiveButton("Delete", (confirm, choice) -> {
                                        try {
                                            new EncryptedMemory(this).forgetPrinciples();
                                            toast("All learned principles deleted");
                                        } catch (Exception error) {
                                            toast("Could not delete memories");
                                        }
                                    }).show())
                    .show();
        } catch (Exception error) {
            toast("Could not read memories: " + error.getClass().getSimpleName());
        }
    }
}
