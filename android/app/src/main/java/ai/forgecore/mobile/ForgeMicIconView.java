package ai.forgecore.mobile;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.View;

/** Minimal white microphone glyph sized to sit inside the purple talk button. */
final class ForgeMicIconView extends View {
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final float density;

    ForgeMicIconView(Context context) {
        super(context);
        density = getResources().getDisplayMetrics().density;
        setContentDescription("Voice chat is not connected yet");
    }

    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float cx = getWidth() * .5f;
        float cy = getHeight() * .46f;
        paint.setColor(Color.WHITE);
        paint.setStrokeWidth(2.3f * density);
        paint.setStrokeCap(Paint.Cap.ROUND);
        paint.setStyle(Paint.Style.STROKE);
        canvas.drawRoundRect(cx - 4.5f * density, cy - 11 * density,
                cx + 4.5f * density, cy + 4 * density,
                5 * density, 5 * density, paint);
        canvas.drawArc(cx - 9 * density, cy - 2 * density,
                cx + 9 * density, cy + 12 * density, 0, 180, false, paint);
        canvas.drawLine(cx, cy + 12 * density, cx, cy + 18 * density, paint);
        canvas.drawLine(cx - 5 * density, cy + 18 * density,
                cx + 5 * density, cy + 18 * density, paint);
        paint.setStyle(Paint.Style.FILL);
    }
}
