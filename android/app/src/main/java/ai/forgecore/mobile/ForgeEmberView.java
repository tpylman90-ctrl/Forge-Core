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
    private volatile long pulseAt = 0L;

    ForgeEmberView(Context context) {
        super(context);
        density = getResources().getDisplayMetrics().density;
        setLayerType(View.LAYER_TYPE_SOFTWARE, null);
        setContentDescription("Forge, a living cloud of smoke and glowing embers");
    }

    void setMode(Mode next) {
        if (next == null || mode == next) return;
        mode = next;
        postInvalidateOnAnimation();
    }

    void pulse() {
        pulseAt = System.currentTimeMillis();
        postInvalidateOnAnimation();
    }

    void setAudioLevel(float rmsDb) {
        audioLevel = Math.max(0f, Math.min(1f, (rmsDb + 8f) / 20f));
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
        Mode currentMode = mode;
        boolean listening = currentMode == Mode.LISTENING;
        boolean active = currentMode == Mode.THINKING || currentMode == Mode.RESPONDING ||
                currentMode == Mode.SPEAKING;
        float voicePulse = listening ? audioLevel : 0f;
        float pulse = Math.max(0f, 1f - (System.currentTimeMillis() - pulseAt) / 850f);
        float energy = Math.max(voicePulse, pulse);
        float ring = Math.min(w * .405f, h * .405f) *
                (1f + (listening ? .045f * voicePulse : active ? .018f * (float) Math.sin(time * 2.1f) : 0f));

        paint.setShader(null);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth((1.25f + energy * .8f) * density);
        paint.setColor(Color.argb((int) (105 + energy * 95), 164, 82, 255));
        canvas.drawCircle(cx, cy, ring, paint);
        paint.setStrokeWidth(2f * density);
        paint.setColor(Color.argb((int) (160 + energy * 85), 186, 107, 255));
        canvas.drawArc(cx - ring, cy - ring, cx + ring, cy + ring,
                -99 + (float) Math.sin(time * (active ? .8f : .25f)) * (listening ? 15f : 8f),
                58 + energy * 20f, false, paint);
        paint.setColor(Color.argb((int) (180 + energy * 75), 196, 130, 255));
        canvas.drawArc(cx - ring, cy - ring, cx + ring, cy + ring,
                53 + (float) Math.sin(time * (active ? .95f : .32f)) * 9,
                37 + energy * 13f, false, paint);

        paint.setStyle(Paint.Style.FILL);
        // Mode changes alter smoke density, spread and turbulence without a rigid avatar silhouette.
        float spread = listening ? 1.10f : currentMode == Mode.THINKING ? .88f
                : currentMode == Mode.RESPONDING || currentMode == Mode.SPEAKING ? 1.11f : 1f;
        float turbulence = listening ? .045f : active ? .052f : .018f;
        for (int i = 0; i < PUFFS.length; i++) {
            float phase = time * (active ? .7f : listening ? .62f : .24f) + i * 1.37f;
            float px = cx + PUFFS[i][0] * w * spread + (float) Math.sin(phase) * w * turbulence;
            float py = cy + PUFFS[i][1] * h * spread + (float) Math.cos(phase * .83f) * h * turbulence;
            float rx = PUFFS[i][2] * w * spread * (.94f + .06f * (float) Math.sin(phase));
            float ry = PUFFS[i][3] * h * spread * (.94f + .06f * (float) Math.cos(phase));
            drawSoftEllipse(canvas, px, py, rx, ry, i % 3 == 0 ? smokeDense : smokeSoft,
                    active ? 245 : listening ? (int) (220 + voicePulse * 35) : 225);
        }

        // Warm inner glow and drifting embers give the cloud its recognizable signature.
        drawSoftEllipse(canvas, cx + w * .16f, cy - h * .005f,
                w * (.19f + energy * .035f), h * (.18f + energy * .035f), warmGlow, 255);
        drawSoftEllipse(canvas, cx - w * .10f, cy + h * .11f,
                w * .16f, h * .14f, warmSoft, 255);

        float emberSpeed = listening ? .75f : currentMode == Mode.THINKING ? .8f
                : currentMode == Mode.RESPONDING || currentMode == Mode.SPEAKING ? 1.05f : .38f;
        for (int i = 0; i < 15; i++) {
            float phase = time * emberSpeed * (.75f + (i % 5) * .09f) + i * 2.09f;
            float orbitX = w * (.13f + (i % 4) * .047f) * spread;
            float orbitY = h * (.12f + (i % 3) * .041f) * spread;
            float ex = cx + (float) Math.sin(phase) * orbitX;
            float ey = cy + (float) Math.cos(phase * .81f + i) * orbitY;
            float emberPulse = .65f + .35f * (float) Math.sin(time * (active ? 3f : 1.8f) + i * 1.7f)
                    + energy * .28f;
            float radius = density * (1.15f + (i % 3) * .55f);
            int alpha = Math.max(0, Math.min(255, Math.round(130 + 105 * emberPulse)));
            paint.setColor(Color.argb(alpha, 255, 145 + (i % 2) * 40, 62));
            canvas.drawCircle(ex, ey, radius * 2.6f, paint);
            paint.setColor(Color.argb(alpha, 255, 208, 143));
            canvas.drawCircle(ex, ey, radius * .85f, paint);
        }

        // A few sparks drift outside the cloud, with slow and deliberately restrained motion.
        for (int i = 0; i < 7; i++) {
            float phase = time * (active ? .35f : .16f) + i * 1.9f;
            float ex = cx + (float) Math.sin(phase + i) * w * (.33f + (i % 2) * .045f);
            float ey = cy + (float) Math.cos(phase * .9f + i) * h * .31f;
            paint.setColor(Color.argb(125, 255, 135, 57));
            canvas.drawCircle(ex, ey, density * (1.1f + (i % 2)), paint);
        }

        postInvalidateDelayed(40);
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
