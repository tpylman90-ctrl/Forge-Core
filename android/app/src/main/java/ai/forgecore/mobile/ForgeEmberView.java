÷v¹çv¶ï¾½ÕÝ4wÞm·]åÎô{×ßÓÍ6}·4package ai.forgecore.mobile;

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
    private final Shader smokeSoft = radial(Color.argb(120, 132, 125, 139));
    private final Shader smokeDense = radial(Color.argb(170, 144, 135, 151));
    private final Shader warmGlow = radial(Color.argb(105, 255, 108, 36));
    private final Shader warmSoft = radial(Color.argb(88, 255, 144, 66));
    private volatile Mode mode = Mode.IDLE;
    private volatile float audioLevel = 0f;
    private volatile float audioTargetLevel = 0f;
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
        float transitionProgress = Math.max(0f, 1f - (System.currentTimeMillis() - modeChangedAt) / 900f);
        float transition = transitionProgress * transitionProgress * (3f - 2f * transitionProgress);
        float breath = .5f + .5f * (float) Math.sin(time * (thinking ? 1.45f : .72f));
        // Response energy follows Forge's full response state, never token arrival timing.
        float responseBreath = .5f + .5f * (float) Math.sin(time * 1.25f - .8f);
        float speechBeat = .5f + .5f * (float) Math.sin(time * 5.1f);
        float energy = listening ? .12f + .54f * voicePulse
                : speaking ? .24f + .24f * speechBeat
                : responding ? .30f + .18f * responseBreath
                : thinking ? .27f + .12f * breath + .10f * transition
                : .06f + .07f * breath;
        float ringMotion = listening ? .020f + .040f * voicePulse
                : speaking ? .016f + .022f * speechBeat
                : responding ? .014f + .020f * responseBreath
                : thinking ? .010f + .014f * breath
                : .005f + .008f * breath;
        float ring = Math.min(w * .405f, h * .405f) * (1f + ringMotion);

        paint.setShader(null);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth((1.75f + energy * 1.7f) * density);
        paint.setColor(Color.argb((int) (140 + energy * 100), 164, 82, 255));
        canvas.drawCircle(cx, cy, ring, paint);
        paint.setStrokeWidth(2.5f * density);
        paint.setColor(Color.argb((int) (190 + energy * 60), 186, 107, 255));
        canvas.drawArc(cx - ring, cy - ring, cx + ring, cy + ring,
                -99 + (float) Math.sin(time * (listening ? .44f : speaking ? .52f : active ? .48f : .14f))
                        * (listening ? 8f : speaking ? 7f : active ? 8f : 4f),
                60 + energy * 18f, false, paint);
        paint.setColor(Color.argb((int) (205 + energy * 50), 196, 130, 255));
        canvas.drawArc(cx - ring, cy - ring, cx + ring, cy + ring,
                53 + (float) Math.sin(time * (listening ? .48f : speaking ? .55f : active ? .46f : .15f)) * 7,
                38 + energy * 10f, false, paint);

        paint.setStyle(Paint.Style.FILL);
        // Mode changes alter smoke density, spread and turbulence without a rigid avatar silhouette.
        float spread = listening ? 1.10f : thinking ? .94f
                : responding ? 1.13f + .018f * responseBreath : speaking ? 1.06f : 1f;
        float turbulence = listening ? .022f + .027f * voicePulse
                : responding ? .028f + .016f * responseBreath
                : speaking ? .024f + .012f * speechBeat
                : thinking ? .024f : .010f;
        float driftSpeed = listening ? .44f + .12f * voicePulse
                : responding ? .52f + .14f * responseBreath
                : speaking ? .46f + .10f * speechBeat
                : thinking ? .32f : .18f;
        for (int i = 0; i < PUFFS.length; i++) {
            float phase = time * driftSpeed + i * 1.37f;
            float px = cx + PUFFS[i][0] * w * spread + (float) Math.sin(phase) * w * turbulence;
            float py = cy + PUFFS[i][1] * h * spread + (float) Math.cos(phase * .83f) * h * turbulence;
            float rx = PUFFS[i][2] * w * spread * (.94f + .06f * (float) Math.sin(phase));
            float ry = PUFFS[i][3] * h * spread * (.94f + .06f * (float) Math.cos(phase));
            drawSoftEllipse(canvas, px, py, rx, ry, i % 3 == 0 ? smokeDense : smokeSoft,
                    listening ? 220 + (int) (voicePulse * 35)
                            : responding ? 224 + (int) (responseBreath * 25)
                            : thinking ? 240 : speaking ? 232 + (int) (speechBeat * 18) : 220);
        }

        // Warm inner glow and drifting embers give the cloud its recognizable signature.
        int coreGlow = listening ? 188 + (int) (voicePulse * 34)
                : responding ? 206 + (int) (responseBreath * 35)
                : thinking ? 198 + (int) (breath * 18)
                : speaking ? 198 + (int) (speechBeat * 24) : 178;
        drawSoftEllipse(canvas, cx + w * .16f, cy - h * .005f,
                w * (.20f + energy * .06f), h * (.19f + energy * .06f), warmGlow, coreGlow);
        drawSoftEllipse(canvas, cx - w * .10f, cy + h * .11f,
                w * (.17f + responseBreath * .018f), h * (.15f + responseBreath * .018f), warmSoft,
                responding ? 195 + (int) (responseBreath * 45) : 210);

        float emberSpeed = listening ? .42f + .20f * voicePulse
                : responding ? .52f + .14f * responseBreath
                : thinking ? .32f : speaking ? .44f + .10f * speechBeat : .22f;
        for (int i = 0; i < 16; i++) {
            float phase = time * emberSpeed * (.75f + (i % 5) * .09f) + i * 2.09f;
            float orbitX = w * (.13f + (i % 4) * .047f) * spread;
            float orbitY = h * (.12f + (i % 3) * .041f) * spread;
            float ex = cx + (float) Math.sin(phase) * orbitX;
            float ey = cy + (float) Math.cos(phase * .81f + i) * orbitY;
            float flickerSpeed = listening ? 1.65f : responding ? 1.95f : speaking ? 2.1f : thinking ? 1.35f : 1.05f;
            float emberPulse = .68f + .25f * (float) Math.sin(time * flickerSpeed + i * 1.7f)
                    + energy * .28f;
            float radius = density * (1.5f + (i % 3) * .65f);
            int alpha = Math.max(0, Math.min(255, Math.round(145 + 100 * emberPulse)));
            paint.setColor(Color.argb(alpha, 255, 145 + (i % 2) * 40, 62));
            canvas.drawCircle(ex, ey, radius * 3.2f, paint);
            paint.setColor(Color.argb(alpha, 255, 208, 143));
            canvas.drawCircle(ex, ey, radius * .85f, paint);
        }

        // A few sparks drift outside the cloud, accelerating as Forge becomes active.
        float sparkSpeed = listening ? .18f + .08f * voicePulse
                : responding ? .28f + .12f * responseBreath
                : speaking ? .24f : thinking ? .18f : .10f;
        for (int i = 0; i < 6; i++) {
            float phase = time * sparkSpeed + i * 1.9f;
            float ex = cx + (float) Math.sin(phase + i) * w * (.33f + (i % 2) * .045f);
            float ey = cy + (float) Math.cos(phase * .9f + i) * h * .31f;
            paint.setColor(Color.argb(responding ? 165 : 135, 255, 135, 57));
            canvas.drawCircle(ex, ey, density * (1.35f + (i % 2) * .55f), paint);
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
