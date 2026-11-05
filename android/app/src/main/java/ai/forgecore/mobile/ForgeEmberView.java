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
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final float density;
    private final long startTime = System.currentTimeMillis();
    private final Shader smokeSoft = radial(Color.argb(32, 142, 132, 145));
    private final Shader smokeDense = radial(Color.argb(42, 142, 132, 145));
    private final Shader warmGlow = radial(Color.argb(35, 255, 108, 36));
    private final Shader warmSoft = radial(Color.argb(25, 255, 144, 66));

    ForgeEmberView(Context context) {
        super(context);
        density = getResources().getDisplayMetrics().density;
        setLayerType(View.LAYER_TYPE_SOFTWARE, null);
        setContentDescription("Forge, a living cloud of smoke and glowing embers");
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
        float ring = Math.min(w * .405f, h * .405f);

        paint.setShader(null);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(1.25f * density);
        paint.setColor(Color.argb(90, 164, 82, 255));
        canvas.drawCircle(cx, cy, ring, paint);
        paint.setStrokeWidth(2f * density);
        paint.setColor(Color.argb(130, 186, 107, 255));
        canvas.drawArc(cx - ring, cy - ring, cx + ring, cy + ring,
                -99 + (float) Math.sin(time * .25f) * 8, 58, false, paint);
        paint.setColor(Color.argb(170, 196, 130, 255));
        canvas.drawArc(cx - ring, cy - ring, cx + ring, cy + ring,
                53 + (float) Math.sin(time * .32f) * 7, 37, false, paint);

        paint.setStyle(Paint.Style.FILL);
        // Overlapping translucent volumes create a soft, irregular smoke silhouette.
        float[][] puffs = {
                {-.40f, .03f, .24f, .27f}, {-.26f, -.16f, .28f, .29f},
                {-.08f, -.26f, .25f, .26f}, {.13f, -.23f, .31f, .30f},
                {.34f, -.09f, .25f, .27f}, {.37f, .12f, .22f, .23f},
                {.18f, .23f, .31f, .25f}, {-.06f, .27f, .29f, .24f},
                {-.31f, .22f, .26f, .24f}, {-.45f, -.04f, .19f, .22f},
                {-.05f, .02f, .35f, .31f}, {.23f, .01f, .32f, .30f}
        };
        for (int i = 0; i < puffs.length; i++) {
            float phase = time * (.24f + (i % 4) * .045f) + i * 1.37f;
            float px = cx + puffs[i][0] * w + (float) Math.sin(phase) * w * .018f;
            float py = cy + puffs[i][1] * h + (float) Math.cos(phase * .83f) * h * .022f;
            float rx = puffs[i][2] * w * (.96f + .04f * (float) Math.sin(phase));
            float ry = puffs[i][3] * h * (.96f + .04f * (float) Math.cos(phase));
            drawSoftEllipse(canvas, px, py, rx, ry,
                    i % 3 == 0 ? smokeDense : smokeSoft);
        }

        // Warm inner glow and drifting embers give the cloud its recognizable signature.
        drawSoftEllipse(canvas, cx + w * .16f, cy - h * .005f,
                w * .19f, h * .18f, warmGlow);
        drawSoftEllipse(canvas, cx - w * .10f, cy + h * .11f,
                w * .16f, h * .14f, warmSoft);

        for (int i = 0; i < 15; i++) {
            float phase = time * (.22f + (i % 5) * .035f) + i * 2.09f;
            float orbitX = w * (.13f + (i % 4) * .047f);
            float orbitY = h * (.12f + (i % 3) * .041f);
            float ex = cx + (float) Math.sin(phase) * orbitX;
            float ey = cy + (float) Math.cos(phase * .81f + i) * orbitY;
            float pulse = .65f + .35f * (float) Math.sin(time * 1.8f + i * 1.7f);
            float radius = density * (1.15f + (i % 3) * .55f);
            int alpha = Math.round(150 + 100 * pulse);
            paint.setColor(Color.argb(alpha, 255, 145 + (i % 2) * 40, 62));
            canvas.drawCircle(ex, ey, radius * 2.6f, paint);
            paint.setColor(Color.argb(alpha, 255, 208, 143));
            canvas.drawCircle(ex, ey, radius * .85f, paint);
        }

        // A few sparks drift outside the cloud, with slow and deliberately restrained motion.
        for (int i = 0; i < 7; i++) {
            float phase = time * .16f + i * 1.9f;
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

    private void drawSoftEllipse(Canvas canvas, float cx, float cy, float rx, float ry, Shader gradient) {
        if (rx <= 0 || ry <= 0) return;
        int save = canvas.save();
        canvas.translate(cx, cy);
        canvas.scale(rx, ry);
        paint.setShader(gradient);
        canvas.drawCircle(0, 0, 1, paint);
        paint.setShader(null);
        canvas.restoreToCount(save);
    }
}
