package ai.forgecore.mobile;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffColorFilter;
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
    private final Bitmap smokeTexture;
    private final PorterDuffColorFilter[] smokeTint = new PorterDuffColorFilter[4];
    private final Shader[] smokeHalo = new Shader[4];
    private final Shader[] nodeGlow = new Shader[4];
    private final Shader[] coreGlow = new Shader[4];
    private final float[] nodeScreenX = new float[NODE_COUNT];
    private final float[] nodeScreenY = new float[NODE_COUNT];
    private final float[] nodeWorldX = new float[NODE_COUNT];
    private final float[] nodeWorldY = new float[NODE_COUNT];
    private final float[] nodeWorldZ = new float[NODE_COUNT];
    private final float[] pointA = new float[3];
    private final float[] pointB = new float[3];
    private volatile Mode mode = Mode.IDLE;
    private volatile float audioLevel = 0f;
    private volatile float audioTargetLevel = 0f;
    private volatile long modeChangedAt = System.currentTimeMillis();
    private volatile Mode previousMode = Mode.IDLE;

    ForgeEmberView(Context context) {
        super(context);
        density = getResources().getDisplayMetrics().density;
        smokeTexture = createSmokeTexture();
        smokeHalo[0] = radial(0xA84A1525);
        smokeHalo[1] = radial(0xA82D0B5A);
        smokeHalo[2] = radial(0xB8752574);
        smokeHalo[3] = radial(0xA8004650);
        for (int i = 0; i < 4; i++) {
            smokeTint[i] = new PorterDuffColorFilter(PALETTE[i][3], PorterDuff.Mode.SRC_IN);
            nodeGlow[i] = radial(withAlpha(PALETTE[i][0], 230));
        }
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

        // Subtle inner haze plus a low-frequency procedural texture give the aura depth.
        float auraScale = listening ? 1.07f + voice * .05f : smelting ? .91f + responseBreath * .025f
                : tempering ? 1.02f : .98f + breath * .035f;
        drawSoftEllipse(canvas, cx, cy, w * .49f * auraScale, h * .46f * auraScale,
                smokeHalo[paletteIndex], listening ? 170 : smelting ? 185 : 145);
        drawSmoke(canvas, cx, cy, w, h, time, auraScale, paletteIndex,
                listening ? 150 + (int) (voice * 55) : smelting ? 185 : 135);

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
            nodeWorldX[i] = xr;
            nodeWorldY[i] = yr;
            nodeWorldZ[i] = zr;
            nodeScreenX[i] = cx + xr * scale;
            nodeScreenY[i] = cy + yr * scale * .88f;
        }

        // Back halves of the gyroscopes sit behind the synaptic core.
        drawGyro(canvas, cx, cy, w, h, time, ringColor, energy, paletteIndex, false);

        // Thin 3D proximity links avoid the tangled false connections of a flat projection.
        paint.setShader(null);
        paint.setStyle(Paint.Style.STROKE);
        int arcBeat = (int) (time * (smelting ? 2.4f : listening ? 1.35f : .55f));
        for (int i = 0; i < NODE_COUNT; i++) {
            for (int j = i + 1; j < NODE_COUNT; j++) {
                float dx = nodeWorldX[i] - nodeWorldX[j];
                float dy = nodeWorldY[i] - nodeWorldY[j];
                float dz = nodeWorldZ[i] - nodeWorldZ[j];
                float distance = (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
                float maxDistance = .79f;
                if (distance >= maxDistance) continue;
                float depth = (nodeWorldZ[i] + nodeWorldZ[j]) * .5f;
                int alpha = Math.max(15, (int) ((1f - distance / maxDistance) * (smelting ? 112 : 76)
                        * (.55f + Math.max(0f, depth) * .45f)));
                paint.setColor(withAlpha(synapseColor, alpha));
                paint.setStrokeWidth((smelting ? .82f : .62f) * density);
                canvas.drawLine(nodeScreenX[i], nodeScreenY[i], nodeScreenX[j], nodeScreenY[j], paint);
                if (smelting && (i * 31 + j * 7 + arcBeat * 3) % 67 == 0) {
                    drawArc(canvas, nodeScreenX[i], nodeScreenY[i], nodeScreenX[j], nodeScreenY[j], hotColor, time, i + j);
                }
            }
        }

        // Render rear nodes first with lower intensity, then front nodes with sharper cores.
        paint.setStyle(Paint.Style.FILL);
        for (int pass = 0; pass < 2; pass++) for (int i = 0; i < NODE_COUNT; i++) {
            boolean front = nodeWorldZ[i] >= 0f;
            if (front != (pass == 1)) continue;
            float pulse = .76f + .24f * (float) Math.sin(time * (smelting ? 2.3f : 1.05f) + i * 1.4f);
            float depthScale = .82f + (nodeWorldZ[i] + 1f) * .12f;
            float radius = density * (1.25f + (i % 4) * .20f) * depthScale;
            int color = smelting && i % 6 == 0 ? Color.WHITE : nodeColor;
            drawSoftEllipse(canvas, nodeScreenX[i], nodeScreenY[i], radius * 2.9f, radius * 2.9f,
                    nodeGlow[paletteIndex], (front ? 175 : 88) + (int) (pulse * 55));
            paint.setColor(withAlpha(color, front ? 220 : 125));
            canvas.drawCircle(nodeScreenX[i], nodeScreenY[i], radius * pulse * .82f, paint);
            paint.setColor(withAlpha(Color.WHITE, front ? 235 : 125));
            canvas.drawCircle(nodeScreenX[i], nodeScreenY[i], Math.max(.75f * density, radius * .30f), paint);
        }

        // Front ring segments catch light across the core; their rear halves stay behind it.
        drawGyro(canvas, cx, cy, w, h, time, ringColor, energy, paletteIndex, true);

        // Hot central point, plus the smooth thermal pulse for the entire processing interval.
        float coreRadius = density * (5f + energy * 2f);
        drawSoftEllipse(canvas, cx, cy, coreRadius * 6.8f, coreRadius * 5.6f, coreGlow[paletteIndex],
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
                          int color, float energy, int paletteIndex, boolean frontHalf) {
        paint.setShader(null);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeCap(Paint.Cap.ROUND);
        float radiusX = Math.min(w * .39f, h * .46f);
        float radiusY = Math.min(h * .35f, w * .40f);
        float spin = paletteIndex == 2 ? 1.0f : paletteIndex == 1 ? .48f : paletteIndex == 3 ? .13f : .18f;
        final int segments = 56;
        for (int ring = 0; ring < 4; ring++) {
            float size = 1f - ring * .105f;
            float direction = ring % 2 == 0 ? 1f : -1f;
            float yaw = time * spin * direction * (1.05f + ring * .21f) + ring * 1.21f;
            float pitch = .62f + (float) Math.sin(time * .23f + ring) * .13f + ring * .12f;
            float roll = time * spin * (ring % 2 == 0 ? .28f : -.21f) + ring * .75f;
            for (int segment = 0; segment < segments; segment++) {
                float a0 = (float) (segment * Math.PI * 2.0 / segments);
                float a1 = (float) ((segment + 1) * Math.PI * 2.0 / segments);
                float c0 = (float) Math.cos(a0), s0 = (float) Math.sin(a0);
                float c1 = (float) Math.cos(a1), s1 = (float) Math.sin(a1);
                float x0, y0, z0, x1, y1, z1;
                if (ring == 0) { x0 = c0; y0 = s0; z0 = 0f; x1 = c1; y1 = s1; z1 = 0f; }
                else if (ring == 1) { x0 = c0; y0 = 0f; z0 = s0; x1 = c1; y1 = 0f; z1 = s1; }
                else if (ring == 2) { x0 = 0f; y0 = c0; z0 = s0; x1 = 0f; y1 = c1; z1 = s1; }
                else {
                    x0 = .707f * c0 - .408f * s0; y0 = .707f * c0 + .408f * s0; z0 = .816f * s0;
                    x1 = .707f * c1 - .408f * s1; y1 = .707f * c1 + .408f * s1; z1 = .816f * s1;
                }
                projectGyroPoint(x0, y0, z0, yaw, pitch, roll, radiusX * size, radiusY * size, pointA);
                projectGyroPoint(x1, y1, z1, yaw, pitch, roll, radiusX * size, radiusY * size, pointB);
                float depth = (pointA[2] + pointB[2]) * .5f;
                if ((depth >= 0f) != frontHalf) continue;
                float light = Math.max(0f, Math.min(1f, (depth + .25f) / 1.25f));
                float xA = cx + pointA[0], yA = cy + pointA[1];
                float xB = cx + pointB[0], yB = cy + pointB[1];
                if (frontHalf) {
                    paint.setStrokeWidth((2.4f + energy * .55f) * density);
                    paint.setColor(withAlpha(color, 20 + (int) (light * 34)));
                    canvas.drawLine(xA, yA, xB, yB, paint);
                    paint.setStrokeWidth((ring == 1 ? 1.2f : .78f) * density);
                    paint.setColor(withAlpha(paletteIndex == 2 && segment % 11 == 0 ? Color.WHITE : color,
                            88 + (int) (light * 122)));
                } else {
                    paint.setStrokeWidth(.68f * density);
                    paint.setColor(withAlpha(color, 23 + (int) (light * 34)));
                }
                canvas.drawLine(xA, yA, xB, yB, paint);
            }
        }
        paint.setStrokeCap(Paint.Cap.BUTT);
        paint.setStyle(Paint.Style.FILL);
    }

    private void drawSmoke(Canvas canvas, float cx, float cy, float w, float h, float time,
                           float scale, int paletteIndex, int opacity) {
        paint.setStyle(Paint.Style.FILL);
        paint.setShader(null);
        paint.setColorFilter(smokeTint[paletteIndex]);
        paint.setAlpha(opacity);
        paint.setFilterBitmap(true);
        oval.set(cx - w * .52f * scale, cy - h * .49f * scale,
                cx + w * .52f * scale, cy + h * .49f * scale);
        int save = canvas.save();
        canvas.rotate((float) Math.sin(time * .16f) * 3.2f + time * (paletteIndex == 2 ? .8f : .25f), cx, cy);
        canvas.drawBitmap(smokeTexture, null, oval, paint);
        canvas.restoreToCount(save);
        // A slower, offset second veil breaks up the silhouette and suggests depth.
        paint.setAlpha(opacity / 2);
        oval.set(cx - w * .44f * scale, cy - h * .42f * scale,
                cx + w * .44f * scale, cy + h * .42f * scale);
        save = canvas.save();
        canvas.rotate(-time * .31f + 27f, cx, cy);
        canvas.drawBitmap(smokeTexture, null, oval, paint);
        canvas.restoreToCount(save);
        paint.setColorFilter(null);
        paint.setAlpha(255);
        paint.setFilterBitmap(false);
    }

    private static Bitmap createSmokeTexture() {
        final int size = 256;
        Bitmap bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[size * size];
        for (int py = 0; py < size; py++) {
            float y = (py + .5f - size * .5f) / (size * .5f);
            for (int px = 0; px < size; px++) {
                float x = (px + .5f - size * .5f) / (size * .5f);
                float radius = (float) Math.sqrt(x * x + y * y);
                float low = valueNoise(x * 3.8f + 8f, y * 3.8f - 4f, 13);
                float mid = valueNoise(x * 9.2f - 3f, y * 9.2f + 7f, 29);
                float high = valueNoise(x * 20f + 5f, y * 20f - 9f, 41);
                float turbulence = low * .58f + mid * .29f + high * .13f;
                float centerRadius = .63f + (low - .5f) * .20f;
                float distance = (radius - centerRadius) / .27f;
                float shell = (float) Math.exp(-distance * distance * 1.15f);
                float wisps = Math.max(0f, Math.min(1f, (turbulence - .20f) * 1.55f));
                float innerHaze = (float) Math.exp(-radius * radius * 5.4f) * .16f;
                float density = Math.min(1f, shell * (.22f + wisps * .78f) + innerHaze);
                // Fade the corners and let fine wisps break up the outer edge.
                density *= Math.max(0f, Math.min(1f, (1.42f - radius) * 2.1f));
                int alpha = Math.max(0, Math.min(210, (int) (density * 210f)));
                pixels[py * size + px] = Color.argb(alpha, 255, 255, 255);
            }
        }
        bitmap.setPixels(pixels, 0, size, 0, 0, size, size);
        return bitmap;
    }

    private static float valueNoise(float x, float y, int seed) {
        int ix = (int) Math.floor(x), iy = (int) Math.floor(y);
        float fx = x - ix, fy = y - iy;
        fx = fx * fx * (3f - 2f * fx);
        fy = fy * fy * (3f - 2f * fy);
        float a = hashNoise(ix, iy, seed), b = hashNoise(ix + 1, iy, seed);
        float c = hashNoise(ix, iy + 1, seed), d = hashNoise(ix + 1, iy + 1, seed);
        float top = a + (b - a) * fx;
        float bottom = c + (d - c) * fx;
        return top + (bottom - top) * fy;
    }

    private static float hashNoise(int x, int y, int seed) {
        int n = x + y * 57 + seed * 131;
        n = (n << 13) ^ n;
        int hash = n * (n * n * 15731 + 789221) + 1376312589;
        return (hash & 0x7fffffff) / 2147483647f;
    }

    private static void projectGyroPoint(float x, float y, float z, float yaw, float pitch, float roll,
                                         float radiusX, float radiusY, float[] out) {
        float cyaw = (float) Math.cos(yaw), syaw = (float) Math.sin(yaw);
        float x1 = x * cyaw - z * syaw;
        float z1 = x * syaw + z * cyaw;
        float cp = (float) Math.cos(pitch), sp = (float) Math.sin(pitch);
        float y2 = y * cp - z1 * sp;
        float z2 = y * sp + z1 * cp;
        float cr = (float) Math.cos(roll), sr = (float) Math.sin(roll);
        float x3 = x1 * cr - y2 * sr;
        float y3 = x1 * sr + y2 * cr;
        float perspective = 1f + z2 * .075f;
        out[0] = x3 * radiusX * perspective;
        out[1] = y3 * radiusY * perspective;
        out[2] = z2;
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
