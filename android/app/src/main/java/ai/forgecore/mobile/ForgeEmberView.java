package ai.forgecore.mobile;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RadialGradient;
import android.graphics.RectF;
import android.graphics.Shader;
import android.view.View;

/** A compact, state-driven mind-vessel visualizer for Forge's chat screen. */
final class ForgeEmberView extends View {
    enum Mode { IDLE, LISTENING, THINKING, RESPONDING, SPEAKING }

    private static final int NODE_COUNT = 26;
    private static final float GOLDEN_ANGLE = 2.3999632f;
    // Node, synapse, gyroscope, and hot-core colors for idle/listen/smelt/temper.
    private static final int[][] PALETTE = {
            {0xFFFF9900, 0xFFCC4400, 0xFFB74C40, 0xFFFF7A26},
            {0xFFE0B0FF, 0xFF8A2BE2, 0xFFB477FF, 0xFFC274FF},
            {0xFFFFFFFF, 0xFFFFB51B, 0xFFFF7A24, 0xFFFF6A28},
            {0xFF00FFFF, 0xFF4B0082, 0xFF23D5D9, 0xFF22C8C4}
    };

    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF oval = new RectF();
    private final float density;
    private final long startTime = System.currentTimeMillis();
    private final Shader[] smokeHalo = new Shader[4];
    private final Shader[] smokePuff = new Shader[4];
    private final Shader[] coreGlow = new Shader[4];
    private final float[] nodeX = new float[NODE_COUNT];
    private final float[] nodeY = new float[NODE_COUNT];
    private final float[] nodeZ = new float[NODE_COUNT];
    private volatile Mode mode = Mode.IDLE;
    private volatile float audioLevel = 0f;
    private volatile float audioTargetLevel = 0f;
    private volatile long modeChangedAt = System.currentTimeMillis();
    private volatile Mode previousMode = Mode.IDLE;

    ForgeEmberView(Context context) {
        super(context);
        density = getResources().getDisplayMetrics().density;
        smokeHalo[0] = radial(0xA84A1525);
        smokeHalo[1] = radial(0xA82D0B5A);
        smokeHalo[2] = radial(0xB8752574);
        smokeHalo[3] = radial(0xA8004650);
        smokePuff[0] = radial(0xB51A0A2A);
        smokePuff[1] = radial(0xB50F0226);
        smokePuff[2] = radial(0xB56A0DAD);
        smokePuff[3] = radial(0xB5120A2A);
        coreGlow[0] = radial(0xD9FF9900);
        coreGlow[1] = radial(0xD98A2BE2);
        coreGlow[2] = radial(0xE8FF8A18);
        coreGlow[3] = radial(0xD900FFFF);
        setContentDescription("Forge, a living neural core within orbital rings and plasma smoke");
    }

    void setMode(Mode next) {
        if (next == null || mode == next) return;
        previousMode = mode;
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
        float w = getWidth(), h = getHeight();
        if (w <= 0 || h <= 0) return;
        float cx = w * .5f, cy = h * .48f;
        float time = (System.currentTimeMillis() - startTime) / 1000f;
        float elapsed = System.currentTimeMillis() - modeChangedAt;
        float transition = Math.min(1f, elapsed / 600f);
        transition = transition * transition * (3f - 2f * transition);
        Mode current = mode;
        int paletteIndex = current == Mode.LISTENING ? 1 : current == Mode.THINKING || current == Mode.RESPONDING ? 2
                : current == Mode.SPEAKING ? 3 : 0;
        int oldIndex = previousMode == Mode.LISTENING ? 1 : previousMode == Mode.THINKING || previousMode == Mode.RESPONDING ? 2
                : previousMode == Mode.SPEAKING ? 3 : 0;
        boolean listening = current == Mode.LISTENING;
        boolean smelting = current == Mode.THINKING || current == Mode.RESPONDING;
        boolean tempering = current == Mode.SPEAKING;
        float audioTarget = audioTargetLevel;
        audioLevel += (audioTarget - audioLevel) * .16f;
        float voice = listening ? audioLevel : 0f;
        float breath = .5f + .5f * (float) Math.sin(time * (listening ? 1.15f : .58f));
        float responseBreath = .5f + .5f * (float) Math.sin(time * 1.18f - .7f);
        float energy = listening ? .34f + voice * .38f : smelting ? .76f + responseBreath * .20f
                : tempering ? .37f : .13f + breath * .08f;
        int nodeColor = blend(PALETTE[oldIndex][0], PALETTE[paletteIndex][0], transition);
        int synapseColor = blend(PALETTE[oldIndex][1], PALETTE[paletteIndex][1], transition);
        int ringColor = blend(PALETTE[oldIndex][2], PALETTE[paletteIndex][2], transition);
        int hotColor = blend(PALETTE[oldIndex][3], PALETTE[paletteIndex][3], transition);

        // Two soft aura layers with a breathing radius. Processing draws the cloud inward.
        float auraScale = listening ? 1.07f + voice * .05f : smelting ? .91f + responseBreath * .025f
                : tempering ? 1.02f : .98f + breath * .035f;
        drawSoftEllipse(canvas, cx, cy, w * .49f * auraScale, h * .46f * auraScale,
                smokeHalo[paletteIndex], listening ? 220 : smelting ? 232 : 205);
        for (int i = 0; i < 13; i++) {
            float angle = i * 2.399f + time * (smelting ? .30f : listening ? .21f : .11f);
            float wobble = (float) Math.sin(time * .72f + i * 1.71f);
            float radius = .24f + (i % 4) * .045f + wobble * (listening ? .025f + voice * .035f : smelting ? -.012f : .014f);
            float px = cx + (float) Math.cos(angle) * w * radius;
            float py = cy + (float) Math.sin(angle) * h * radius * .83f;
            float puff = density * (24f + (i % 3) * 7f);
            drawSoftEllipse(canvas, px, py, puff * 1.25f, puff, smokePuff[paletteIndex],
                    listening ? 190 : smelting ? 205 : 175);
        }

        // Independent tilted ellipses create a lightweight projected gyro vessel.
        drawGyro(canvas, cx, cy, w, h, time, ringColor, energy, paletteIndex);

        // Neural points arranged on a spherical shell and projected into the vessel.
        float yaw = time * (smelting ? 1.12f : listening ? .54f : tempering ? .16f : .22f);
        float pitch = (float) Math.sin(time * .31f) * (smelting ? .30f : .18f);
        float cp = (float) Math.cos(pitch), sp = (float) Math.sin(pitch);
        for (int i = 0; i < NODE_COUNT; i++) {
            float z = 1f - 2f * (i + .5f) / NODE_COUNT;
            float r = (float) Math.sqrt(Math.max(0f, 1f - z * z));
            float theta = i * GOLDEN_ANGLE;
            float x = r * (float) Math.cos(theta);
            float y = r * (float) Math.sin(theta);
            if (listening) {
                x = -1f + 2f * i / (NODE_COUNT - 1f);
                y = (float) Math.sin(i * .72f + time * 2.0f) * (.12f + voice * .30f);
                z *= .48f;
            } else if (!smelting && !tempering) {
                x += (float) Math.sin(time * .37f + i * 1.7f) * .035f;
                y += (float) Math.cos(time * .33f + i * 1.1f) * .035f;
            }
            float xr = x * (float) Math.cos(yaw) - z * (float) Math.sin(yaw);
            float zr = x * (float) Math.sin(yaw) + z * (float) Math.cos(yaw);
            float yr = y * cp - zr * sp;
            zr = y * sp + zr * cp;
            float scale = .31f * Math.min(w, h) * (1f + zr * .12f);
            nodeX[i] = cx + xr * scale;
            nodeY[i] = cy + yr * scale * .88f;
            nodeZ[i] = zr;
        }

        // Thin proximity links; the active response lights the whole connected core.
        paint.setShader(null);
        paint.setStyle(Paint.Style.STROKE);
        int arcBeat = (int) (time * (smelting ? 2.4f : listening ? 1.35f : .55f));
        for (int i = 0; i < NODE_COUNT; i++) {
            for (int j = i + 1; j < NODE_COUNT; j++) {
                float dx = nodeX[i] - nodeX[j], dy = nodeY[i] - nodeY[j];
                float distance = (float) Math.sqrt(dx * dx + dy * dy);
                float maxDistance = Math.min(w, h) * .205f;
                if (distance >= maxDistance) continue;
                int alpha = Math.max(24, (int) ((1f - distance / maxDistance) * (smelting ? 190 : 125)));
                paint.setColor(withAlpha(synapseColor, alpha));
                paint.setStrokeWidth((smelting ? 1.15f : .85f) * density);
                canvas.drawLine(nodeX[i], nodeY[i], nodeX[j], nodeY[j], paint);
                if (smelting && (i * 31 + j * 7 + arcBeat * 3) % 29 == 0) {
                    drawArc(canvas, nodeX[i], nodeY[i], nodeX[j], nodeY[j], hotColor, time, i + j);
                }
            }
        }

        // Nodes have a soft halo and a bright pin-point center.
        paint.setStyle(Paint.Style.FILL);
        for (int i = 0; i < NODE_COUNT; i++) {
            float pulse = .76f + .24f * (float) Math.sin(time * (smelting ? 2.3f : 1.05f) + i * 1.4f);
            float radius = density * (2.0f + (i % 4) * .35f) * (smelting ? 1.13f : 1f);
            int color = smelting && i % 4 == 0 ? Color.WHITE : nodeColor;
            paint.setColor(withAlpha(color, (int) (150 + 100 * pulse)));
            canvas.drawCircle(nodeX[i], nodeY[i], radius * 3.6f, paint);
            paint.setColor(color);
            canvas.drawCircle(nodeX[i], nodeY[i], radius * pulse, paint);
            paint.setColor(Color.WHITE);
            canvas.drawCircle(nodeX[i], nodeY[i], Math.max(1f, radius * .34f), paint);
        }

        // Hot central point, plus the smooth thermal pulse for the entire processing interval.
        float coreRadius = density * (8f + energy * 5f);
        drawSoftEllipse(canvas, cx, cy, coreRadius * 5f, coreRadius * 4.1f, coreGlow[paletteIndex],
                smelting ? 255 : listening ? 215 : 190);
        paint.setShader(null);
        paint.setStyle(Paint.Style.FILL);
        paint.setColor(hotColor);
        canvas.drawCircle(cx, cy, coreRadius, paint);
        paint.setColor(Color.WHITE);
        canvas.drawCircle(cx, cy, coreRadius * .36f, paint);

        // Sparks skim outward. Listening sends a few visible streams from the lower pole.
        drawSparks(canvas, cx, cy, w, h, time, hotColor, smelting, tempering);
        if (listening) drawInputStreams(canvas, cx, cy, w, h, time, voice, nodeColor);

        postInvalidateOnAnimation();
    }

    private void drawGyro(Canvas canvas, float cx, float cy, float w, float h, float time,
                          int color, float energy, int paletteIndex) {
        paint.setShader(null);
        paint.setStyle(Paint.Style.STROKE);
        float baseW = Math.min(w * .86f, h * 1.18f);
        float baseH = Math.min(h * .86f, w * .72f);
        float spin = paletteIndex == 2 ? 1.0f : paletteIndex == 1 ? .54f : paletteIndex == 3 ? .16f : .21f;
        for (int i = 0; i < 4; i++) {
            int save = canvas.save();
            canvas.translate(cx, cy);
            canvas.rotate((i % 2 == 0 ? 1f : -1f) * time * spin * (18f + i * 4f) + i * 47f);
            float tilt = i == 0 ? .40f : i == 1 ? .67f : i == 2 ? 1f : .52f;
            canvas.scale(1f, tilt);
            float size = 1f - i * .115f;
            oval.set(-baseW * size * .5f, -baseH * size * .5f,
                    baseW * size * .5f, baseH * size * .5f);
            int alpha = i == 0 ? 118 : i == 1 ? 170 : i == 2 ? 112 : 145;
            paint.setStrokeWidth((i == 1 ? 1.9f : 1.25f) * density + energy * .55f * density);
            paint.setColor(withAlpha(color, alpha));
            canvas.drawOval(oval, paint);
            paint.setStrokeWidth(2.6f * density);
            paint.setColor(withAlpha(paletteIndex == 2 ? Color.WHITE : color, 205));
            canvas.drawArc(oval, -76f + i * 62f, 42f + energy * 15f, false, paint);
            if (paletteIndex == 2 && i < 3) {
                paint.setStrokeWidth(1f * density);
                paint.setColor(withAlpha(0xFFFFC45A, 110));
                canvas.drawArc(oval, 90f + i * 28f, 25f, false, paint);
            }
            canvas.restoreToCount(save);
        }
    }

    private void drawArc(Canvas canvas, float x1, float y1, float x2, float y2,
                         int color, float time, int seed) {
        float dx = x2 - x1, dy = y2 - y1;
        float length = (float) Math.sqrt(dx * dx + dy * dy);
        if (length < density) return;
        float nx = -dy / length, ny = dx / length;
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(1.4f * density);
        paint.setColor(withAlpha(color, 220));
        float lastX = x1, lastY = y1;
        for (int s = 1; s <= 5; s++) {
            float t = s / 5f;
            float jitter = s == 5 ? 0f : (float) Math.sin(time * 21f + seed * 4.3f + s * 7f) * 4f * density;
            float nextX = x1 + dx * t + nx * jitter;
            float nextY = y1 + dy * t + ny * jitter;
            canvas.drawLine(lastX, lastY, nextX, nextY, paint);
            lastX = nextX;
            lastY = nextY;
        }
        paint.setStyle(Paint.Style.FILL);
    }

    private void drawSparks(Canvas canvas, float cx, float cy, float w, float h, float time,
                            int color, boolean smelting, boolean tempering) {
        paint.setShader(null);
        paint.setStyle(Paint.Style.FILL);
        int count = smelting ? 17 : 11;
        for (int i = 0; i < count; i++) {
            float phase = time * (smelting ? .82f : tempering ? .18f : .34f) + i * 2.17f;
            float orbit = .34f + (i % 4) * .026f;
            float x = cx + (float) Math.cos(phase) * w * orbit;
            float y = cy + (float) Math.sin(phase * .88f + i) * h * (.29f + (i % 3) * .025f);
            float radius = density * (i % 5 == 0 ? 2.5f : 1.15f);
            paint.setColor(withAlpha(color, smelting ? 175 : 105));
            canvas.drawCircle(x, y, radius * 2.7f, paint);
            paint.setColor(withAlpha(Color.WHITE, smelting && i % 5 == 0 ? 220 : 125));
            canvas.drawCircle(x, y, radius * .55f, paint);
        }
    }

    private void drawInputStreams(Canvas canvas, float cx, float cy, float w, float h,
                                  float time, float voice, int color) {
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(1.3f * density);
        for (int i = 0; i < 3; i++) {
            float phase = time * (1.3f + voice * 1.4f) + i * 2.1f;
            float x = cx + (float) Math.sin(phase) * w * .045f;
            float y = cy + h * (.43f - (phase % 1.6f) * .28f);
            paint.setColor(withAlpha(color, 100 + (int) (voice * 130)));
            canvas.drawLine(x, y + h * .08f, x - (float) Math.sin(phase * .7f) * w * .025f, y, paint);
            paint.setStyle(Paint.Style.FILL);
            canvas.drawCircle(x, y, density * (1.8f + voice * 2f), paint);
            paint.setStyle(Paint.Style.STROKE);
        }
        paint.setStyle(Paint.Style.FILL);
    }

    private void drawSoftEllipse(Canvas canvas, float cx, float cy, float rx, float ry,
                                 Shader gradient, int opacity) {
        if (rx <= 0 || ry <= 0) return;
        int save = canvas.save();
        canvas.translate(cx, cy);
        canvas.scale(rx, ry);
        paint.setStyle(Paint.Style.FILL);
        paint.setShader(gradient);
        paint.setAlpha(opacity);
        canvas.drawCircle(0, 0, 1, paint);
        paint.setAlpha(255);
        paint.setShader(null);
        canvas.restoreToCount(save);
    }

    private static Shader radial(int color) {
        int alpha = Color.alpha(color);
        int core = Color.argb(alpha, Color.red(color), Color.green(color), Color.blue(color));
        int mid = Color.argb(alpha / 2, Color.red(color), Color.green(color), Color.blue(color));
        return new RadialGradient(0, 0, 1,
                new int[]{core, mid, Color.TRANSPARENT}, new float[]{0f, .58f, 1f},
                Shader.TileMode.CLAMP);
    }

    private static int withAlpha(int color, int alpha) {
        return Color.argb(Math.max(0, Math.min(255, alpha)), Color.red(color), Color.green(color), Color.blue(color));
    }

    private static int blend(int from, int to, float amount) {
        int a = (int) (Color.alpha(from) + (Color.alpha(to) - Color.alpha(from)) * amount);
        int r = (int) (Color.red(from) + (Color.red(to) - Color.red(from)) * amount);
        int g = (int) (Color.green(from) + (Color.green(to) - Color.green(from)) * amount);
        int b = (int) (Color.blue(from) + (Color.blue(to) - Color.blue(from)) * amount);
        return Color.argb(a, r, g, b);
    }
}
