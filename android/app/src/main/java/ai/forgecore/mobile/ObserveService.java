package ai.forgecore.mobile;

import android.app.AppOpsManager;
import android.app.KeyguardManager;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.app.usage.UsageEvents;
import android.app.usage.UsageStatsManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.Process;

import java.util.Set;

/**
 * Clearly visible, user-started, user-stoppable observation.
 * Only allowlisted package transitions; no screens, texts or app contents.
 */
public final class ObserveService extends Service {
    public static final String STOP = "ai.forgecore.mobile.STOP";
    private static final String STATUS_CHANNEL = "forge_observation";
    private static final String QUESTION_CHANNEL = "forge_questions";
    private static final int STATUS_ID = 101;
    private static final int QUESTION_ID = 102;
    private final Handler handler = new Handler(Looper.getMainLooper());
    private long lastChecked;
    private long lastAskedAt;
    private String lastAskedPackage = "";
    private boolean started = false;

    private final Runnable poll = new Runnable() {
        @Override public void run() {
            if (!started) return;
            try { observeAppSwitch(); }
            catch (Exception ignored) {
                // No external logging of observed app names or personal data.
            }
            handler.postDelayed(this, 9000);
        }
    };

    @Override public void onCreate() {
        super.onCreate();
        NotificationManager manager = getSystemService(NotificationManager.class);
        manager.createNotificationChannel(new NotificationChannel(
                STATUS_CHANNEL, "Forge observation status", NotificationManager.IMPORTANCE_LOW));
        manager.createNotificationChannel(new NotificationChannel(
                QUESTION_CHANNEL, "Forge learning questions", NotificationManager.IMPORTANCE_DEFAULT));
    }

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && STOP.equals(intent.getAction())) {
            stopSelf();
            return START_NOT_STICKY;
        }
        if (started) return START_NOT_STICKY;
        AppOpsManager ops = (AppOpsManager) getSystemService(Context.APP_OPS_SERVICE);
        if (ops.checkOpNoThrow(AppOpsManager.OPSTR_GET_USAGE_STATS, Process.myUid(), getPackageName())
                != AppOpsManager.MODE_ALLOWED) {
            stopSelf();
            return START_NOT_STICKY;
        }
        try {
            if (new EncryptedMemory(this).allowed().isEmpty()) {
                stopSelf();
                return START_NOT_STICKY;
            }
        } catch (Exception error) {
            stopSelf();
            return START_NOT_STICKY;
        }

        Intent stopIntent = new Intent(this, ObserveService.class).setAction(STOP);
        PendingIntent stop = PendingIntent.getService(this, 4, stopIntent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        Notification ongoing = new Notification.Builder(this, STATUS_CHANNEL)
                .setSmallIcon(android.R.drawable.ic_dialog_info)
                .setContentTitle("Forge Core: observation on")
                .setContentText("Only apps you selected. Tap Stop to pause.")
                .setOngoing(true)
                .addAction(android.R.drawable.ic_media_pause, "Stop observing", stop)
                .build();

        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(STATUS_ID, ongoing, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
        } else {
            startForeground(STATUS_ID, ongoing);
        }
        started = true;
        lastChecked = System.currentTimeMillis();
        handler.postDelayed(poll, 2000);
        return START_NOT_STICKY;
    }

    private void observeAppSwitch() throws Exception {
        AppOpsManager ops = (AppOpsManager) getSystemService(Context.APP_OPS_SERVICE);
        if (ops.checkOpNoThrow(AppOpsManager.OPSTR_GET_USAGE_STATS, Process.myUid(), getPackageName())
                != AppOpsManager.MODE_ALLOWED) {
            stopSelf();
            return;
        }
        KeyguardManager lock = (KeyguardManager) getSystemService(Context.KEYGUARD_SERVICE);
        if (lock != null && lock.isDeviceLocked()) {
            lastChecked = System.currentTimeMillis();
            return;
        }
        long now = System.currentTimeMillis();
        UsageStatsManager manager = (UsageStatsManager) getSystemService(Context.USAGE_STATS_SERVICE);
        UsageEvents events = manager.queryEvents(lastChecked, now);
        lastChecked = now;
        if (events == null) return;

        String newest = null;
        UsageEvents.Event event = new UsageEvents.Event();
        while (events.hasNextEvent()) {
            events.getNextEvent(event);
            if (event.getEventType() != UsageEvents.Event.ACTIVITY_RESUMED) continue;
            String name = event.getPackageName();
            if (name == null || name.equals(getPackageName())) continue;
            newest = name;
        }
        if (newest == null) return;
        Set<String> allowed = new EncryptedMemory(this).allowed();
        if (!allowed.contains(newest)) return;
        if (newest.equals(lastAskedPackage) && now - lastAskedAt < 120000) return;
        lastAskedPackage = newest;
        lastAskedAt = now;
        askWhy(newest);
    }

    private void askWhy(String packageName) {
        Intent open = new Intent(this, MainActivity.class);
        open.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP
                | Intent.FLAG_ACTIVITY_SINGLE_TOP);
        open.putExtra("forge_question_app", packageName);
        PendingIntent tap = PendingIntent.getActivity(this, 3, open,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        Notification question = new Notification.Builder(this, QUESTION_CHANNEL)
                .setSmallIcon(android.R.drawable.ic_dialog_info)
                .setContentTitle("Forge has a question")
                .setContentText("What were you trying to accomplish?")
                .setContentIntent(tap)
                .setAutoCancel(true)
                .build();
        getSystemService(NotificationManager.class).notify(QUESTION_ID, question);
    }

    @Override public void onDestroy() {
        started = false;
        handler.removeCallbacks(poll);
        super.onDestroy();
    }

    @Override public IBinder onBind(Intent intent) { return null; }
}
