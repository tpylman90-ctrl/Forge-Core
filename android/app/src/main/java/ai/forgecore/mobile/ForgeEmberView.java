package ai.forgecore.mobile;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RadialGradient;
import android.graphics.Shader;
import android.view.View;

/** A lightweight animated smoke-and-ember presence for Forge's chat screen. */
final class ForgeEmberView extends View {
    enum Mode { IDLE, LISTENING, THINKING, RESPONDING, SPEAKING }

    private static final float[][] PUFFS = {
            {-.40f, .03f, .24f, .27f}, {-.26f, -.16f, .28f, .29f},
            {-.08f, -.26f, .25f, .26f}, {.13f, -.23f, .31f, .30f},
            {.34f, -.09f, .25f, .27f}, {.37f, .12f, .22f, .23f},
            {.18f, .23f, .31f, .25f}, {-.06f, .27f, .29f, .24f},
            {-.31f, .22f, .26f, .24f}, {-.45f, -.04f, .19f, .22f},
            {-.05f, .02f, .35f, .31f}, {.23f, .01f, .32f, .30f}
    };
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final float density;
    private final long startTime = System.currentTimeMillis();
    private final Shader smokeSoft = radial(Color.argb(100, 132, 125, 139));
    private final Shader smokeDense = radial(Color.argb(145, 144, 135, 151));
    private final Shader warmGlow = radial(Color.argb(82, 255, 108, 36));
    private final Shader warmSoft = radial(Color.argb(62, 255, 144, 66));
    private volatile Mode mode = Mode.IDLE;
    private volatile float audioLevel = 0f;
    private volatile float audioTargetLevel = 0f;
    private volatile long pulseAt = 0L;
    private volatile long modeChangedAt = System.currentTimeMillis();

    ForgeEmberView(Context context) {
        super(context);
        density = getResources().getDisplayMetrics().density;
        setContentDescription("Forge, a living cloud of smoke and glowing embers");
    }

    void setMode(Mode next) {
        if (next == null || mode == next) return;
        mode = next;
        modeChangedAt = System.currentTimeMillis();
        pulseAt = modeChangedAt;
        postInvalidateOnAnimation();
    }

    void pulse() {
        pulseAt = System.currentTimeMillis();
        postInvalidateOnAnimation();
    }

    void setAudioLevel(float rmsDb) {
        audioTargetLevel = Math.max(0f, Math.min(1f, (rmsDb + 8f) / 20f));
        postInvalidateOnAnimation();
    }

    @Override protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        postInvalidateOnAnimation();
    }

    @Override protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
    }

    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float w = getWidth();
        float h = getHeight();
        float cx = w * .5f;
        float cy = h * .49f;
        float time = (System.currentTimeMillis() - startTime) / 1000f;
        float audioTarget = audioTargetLevel;
        audioLevel += (audioTarget - audioLevel) * .18f;
        Mode currentMode = mode;
        boolean listening = currentMode == Mode.LISTENING;
        boolean active = currentMode == Mode.THINKING || currentMode == Mode.RESPONDING ||
                currentMode == Mode.SPEAKING;
        boolean thinking = currentMode == Mode.THINKING;
        boolean responding = currentMode == Mode.RESPONDING;
        boolean speaking = currentMode == Mode.SPEAKING;
        float voicePulse = listening ? audioLevel : 0f;
        float pulseProgress = Math.max(0f, 1f - (System.currentTimeMillis() - pulseAt) / 1150f);
        float pulse = pulseProgress * pulseProgress * (3f - 2f * pulseProgress);
        float transitionProgress = Math.max(0f, 1f - (System.currentTimeMillis() - modeChangedAt) / 900f);
        float transition = transitionProgress * transitionProgress * (3f - 2f * transitionProgress);
        float breath = .5f + .5f * (float) Math.sin(time * (thinking ? 1.45f : .72f));
        float speechBeat = .5f + .5f * (float) Math.sin(time * 5.1f);
        float energy = listening ? .12f + .54f * voicePulse
                : speaking ? .18f + .24f * speechBeat
                : responding ? .12f + .56f * pulse
                : thinking ? .17f + .13f * breath + .15f * transition
                : .04f + .05f * breath;
        float ringMotion = listening ? .016f + .032f * voicePulse
                : speaking ? .010f + .018f * speechBeat
                : responding ? .006f + .024f * pulse
                : thinking ? .005f + .012f * breath
                : .004f + .006f * breath;
        float ring = Math.min(w * .405f, h * .405f) * (1f + ringMotion);

        paint.setShader(null);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth((1.25f + energy * .8f) * density);
        paint.setColor(Color.argb((int) (105 + energy * 95), 164, 82, 255));
        canvas.drawCircle(cx, cy, ring, paint);
        paint.setStrokeWidth(2f * density);
        paint.setColor(Color.argb((int) (160 + energy * 85), 186, 107, 255));
        canvas.drawArc(cx - ring, cy - ring, cx + ring, cy + ring,
                -99 + (float) Math.sin(time * (listening ? .44f : speaking ? .52f : active ? .32f : .14f))
                        * (listening ? 8f : speaking ? 6f : 4f),
                54 + energy * 15f, false, paint);
        paint.setColor(Color.argb((int) (180 + energy * 75), 196, 130, 255));
        canvas.drawArc(cx - ring, cy - ring, cx + ring, cy + ring,
                53 + (float) Math.sin(time * (listening ? .48f : speaking ? .55f : active ? .32f : .15f)) * 5,
                32 + energy * 8f, false, paint);

        paint.setStyle(Paint.Style.FILL);
        // Mode changes alter smoke density, spread and turbulence without a rigid avatar silhouette.
        float spread = listening ? 1.10f : thinking ? .88f
                : responding ? 1.12f : speaking ? 1.045f : 1f;
        float turbulence = listening ? .018f + .020f * voicePulse
                : responding ? .024f + .012f * pulse
                : speaking ? .020f + .008f * speechBeat
                : thinking ? .018f : .010f;
        float driftSpeed = listening ? .44f + .12f * voicePulse
                : responding ? .48f + .18f * pulse
                : speaking ? .43f + .08f * speechBeat
                : thinking ? .26f : .18f;
        for (int i = 0; i < PUFFS.length; i++) {
            float phase = time * driftSpeed + i * 1.37f;
            float px = cx + PUFFS[i][0] * w * spread + (float) Math.sin(phase) * w * turbulence;
            float py = cy + PUFFS[i][1] * h * spread + (float) Math.cos(phase * .83f) * h * turbulence;
            float rx = PUFFS[i][2] * w * spread * (.94f + .06f * (float) Math.sin(phase));
            float ry = PUFFS[i][3] * h * spread * (.94f + .06f * (float) Math.cos(phase));
            drawSoftEllipse(canvas, px, py, rx, ry, i % 3 == 0 ? smokeDense : smokeSoft,
                    listening ? 220 + (int) (voicePulse * 35)
                            : responding ? 215 + (int) (pulse * 40)
                            : thinking ? 232 : speaking ? 226 + (int) (speechBeat * 22) : 215);
        }

        // Warm inner glow and drifting embers give the cloud its recognizable signature.
        int coreGlow = listening ? 188 + (int) (voicePulse * 34)
                : responding ? 184 + (int) (pulse * 38)
                : thinking ? 178 + (int) (breath * 18)
                : speaking ? 186 + (int) (speechBeat * 26) : 170;
        drawSoftEllipse(canvas, cx + w * .16f, cy - h * .005f,
                w * (.19f + energy * .045f), h * (.18f + energy * .045f), warmGlow, coreGlow);
        drawSoftEllipse(canvas, cx - w * .10f, cy + h * .11f,
                w * (.16f + pulse * .025f), h * (.14f + pulse * .025f), warmSoft,
                responding ? 180 + (int) (pulse * 70) : 205);

        float emberSpeed = listening ? .42f + .20f * voicePulse
                : responding ? .52f + .28f * pulse
                : thinking ? .32f : speaking ? .44f + .10f * speechBeat : .22f;
        for (int i = 0; i < 12; i++) {
            float phase = time * emberSpeed * (.75f + (i % 5) * .09f) + i * 2.09f;
            float orbitX = w * (.13f + (i % 4) * .047f) * spread;
            float orbitY = h * (.12f + (i % 3) * .041f) * spread;
            float ex = cx + (float) Math.sin(phase) * orbitX;
            float ey = cy + (float) Math.cos(phase * .81f + i) * orbitY;
            float flickerSpeed = listening ? 1.65f : responding ? 1.95f : speaking ? 2.1f : thinking ? 1.35f : 1.05f;
            float emberPulse = .68f + .25f * (float) Math.sin(time * flickerSpeed + i * 1.7f)
                    + energy * .20f;
            float radius = density * (1.15f + (i % 3) * .55f);
            int alpha = Math.max(0, Math.min(255, Math.round(130 + 105 * emberPulse)));
            paint.setColor(Color.argb(alpha, 255, 145 + (i % 2) * 40, 62));
            canvas.drawCircle(ex, ey, radius * 2.6f, paint);
            paint.setColor(Color.argb(alpha, 255, 208, 143));
            canvas.drawCircle(ex, ey, radius * .85f, paint);
        }

        // A few sparks drift outside the cloud, accelerating as Forge becomes active.
        float sparkSpeed = listening ? .18f + .08f * voicePulse
                : responding ? .20f + .11f * pulse
                : speaking ? .20f : thinking ? .14f : .10f;
        for (int i = 0; i < 4; i++) {
            float phase = time * sparkSpeed + i * 1.9f;
            float ex = cx + (float) Math.sin(phase + i) * w * (.33f + (i % 2) * .045f);
            float ey = cy + (float) Math.cos(phase * .9f + i) * h * .31f;
            paint.setColor(Color.argb(125, 255, 135, 57));
            canvas.drawCircle(ex, ey, density * (1.1f + (i % 2)), paint);
        }

        postInvalidateOnAnimation();
    }

    private static Shader radial(int color) {
        int alpha = Color.alpha(color);
        int core = Color.argb(alpha, Color.red(color), Color.green(color), Color.blue(color));
        int mid = Color.argb(alpha / 2, Color.red(color), Color.green(color), Color.blue(color));
        return new RadialGradient(0, 0, 1,
                new int[]{core, mid, Color.TRANSPARENT}, new float[]{0f, .58f, 1f},
                Shader.TileMode.CLAMP);
    }

    private void drawSoftEllipse(Canvas canvas, float cx, float cy, float rx, float ry,
                                 Shader gradient, int opacity) {
        if (rx <= 0 || ry <= 0) return;
        int save = canvas.save();
        canvas.translate(cx, cy);
        canvas.scale(rx, ry);
        paint.setShader(gradient);
        paint.setAlpha(opacity);
        canvas.drawCircle(0, 0, 1, paint);
        paint.setAlpha(255);
        paint.setShader(null);
        canvas.restoreToCount(save);
    }
}
