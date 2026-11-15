package ai.forgecore.mobile;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.View;

/** Quiet waveform and concentric light around the voice control. */
final class ForgeMicAuraView extends View {
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final float density;
    private volatile boolean active;
    private final long startedAt = System.currentTimeMillis();

    ForgeMicAuraView(Context context) {
        super(context);
        density = getResources().getDisplayMetrics().density;
        setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);
    }

    void setActive(boolean value) {
        if (active == value) return;
        active = value;
        postInvalidateOnAnimation();
    }

    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float cx = getWidth() * .5f;
        float cy = getHeight() * .5f;
        float time = (System.currentTimeMillis() - startedAt) / 1000f;
        int color = active ? 0xFFFF9A4D : 0xFFB269FF;
        float breath = active ? .78f + .22f * (float) Math.sin(time * 2.1f) : .72f;
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(1.25f * density);
        paint.setColor(withAlpha(color, active ? 176 : 110));
        canvas.drawCircle(cx, cy, 34f * density, paint);
        paint.setColor(withAlpha(color, active ? 116 : 72));
        canvas.drawCircle(cx, cy, 39f * density, paint);

        paint.setStyle(Paint.Style.FILL);
        int bars = 48;
        for (int i = 0; i < bars; i++) {
            float centered = (i - (bars - 1) * .5f) / ((bars - 1) * .5f);
            float envelope = (float) Math.exp(-centered * centered * 2.35f);
            float wave = .28f + .72f * (float) Math.abs(Math.sin(i * 1.29f + (active ? time * 3.4f : 0f)));
            float amplitude = (2f + 15f * envelope * wave * breath) * density;
            float x = cx + centered * 75f * density;
            paint.setColor(withAlpha(color, (int) ((active ? 225 : 145) * envelope)));
            canvas.drawRoundRect(x - .95f * density, cy - amplitude,
                    x + .95f * density, cy + amplitude, density, density, paint);
        }
        if (active) postInvalidateOnAnimation();
    }

    private static int withAlpha(int color, int alpha) {
        return Color.argb(Math.max(0, Math.min(255, alpha)),
                Color.red(color), Color.green(color), Color.blue(color));
    }
}
