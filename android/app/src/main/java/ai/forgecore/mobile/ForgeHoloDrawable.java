package ai.forgecore.mobile;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorFilter;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Shader;
import android.graphics.drawable.Drawable;

/** A low-noise glass-and-light treatment for Forge interface panels. */
final class ForgeHoloDrawable extends Drawable {
    private final float density;
    private final int accent;
    private final int secondary;
    private int alpha = 255;
    private ColorFilter colorFilter;

    ForgeHoloDrawable(float density, int accent, int secondary) {
        this.density = density;
        this.accent = accent;
        this.secondary = secondary;
    }

    private int dp(float value) { return Math.max(1, Math.round(value * density)); }

    private int withAlpha(int color, int amount) {
        return Color.argb(Math.round(Color.alpha(color) * (amount / 255f) * (alpha / 255f)),
                Color.red(color), Color.green(color), Color.blue(color));
    }

    private Paint paint(int color, float width, Paint.Style style) {
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        paint.setColor(withAlpha(color, Color.alpha(color)));
        paint.setStrokeWidth(width);
        paint.setStyle(style);
        paint.setStrokeCap(Paint.Cap.ROUND);
        paint.setStrokeJoin(Paint.Join.ROUND);
        paint.setColorFilter(colorFilter);
        return paint;
    }

    @Override public void draw(Canvas canvas) {
        Rect b = getBounds();
        float inset = dp(1.2f);
        RectF box = new RectF(b.left + inset, b.top + inset, b.right - inset, b.bottom - inset);
        float radius = dp(18);
        Path clip = new Path();
        clip.addRoundRect(box, radius, radius, Path.Direction.CW);

        Paint fill = paint(Color.WHITE, 0, Paint.Style.FILL);
        fill.setShader(new LinearGradient(box.left, box.top, box.right, box.bottom,
                new int[]{0xF8141B2A, 0xF20A111D, 0xF5131925}, null, Shader.TileMode.CLAMP));
        canvas.drawRoundRect(box, radius, radius, fill);

        int save = canvas.save();
        canvas.clipPath(clip);
        float step = dp(22);
        Paint grid = paint(accent, dp(0.55f), Paint.Style.STROKE);
        grid.setAlpha(withAlpha(accent, 19) >>> 24);
        for (float x = box.left + step; x < box.right; x += step) canvas.drawLine(x, box.top, x, box.bottom, grid);
        for (float y = box.top + step; y < box.bottom; y += step) canvas.drawLine(box.left, y, box.right, y, grid);

        Paint scan = paint(0xFF8DEEFF, dp(0.55f), Paint.Style.STROKE);
        scan.setAlpha(withAlpha(0xFF8DEEFF, 8) >>> 24);
        for (float y = box.top + dp(4); y < box.bottom; y += dp(5)) canvas.drawLine(box.left, y, box.right, y, scan);

        Paint flecks = paint(0xFFD7F9FF, dp(1.1f), Paint.Style.FILL);
        flecks.setAlpha(withAlpha(0xFFD7F9FF, 92) >>> 24);
        float[][] points = {{.18f,.21f},{.79f,.17f},{.68f,.43f},{.27f,.72f},{.86f,.78f},{.54f,.88f}};
        for (float[] p : points) canvas.drawCircle(box.left + box.width()*p[0], box.top + box.height()*p[1], dp(0.65f), flecks);

        Paint wash = paint(accent, dp(1), Paint.Style.STROKE);
        wash.setAlpha(withAlpha(accent, 25) >>> 24);
        canvas.drawLine(box.left + radius, box.top + dp(1), box.right - radius, box.top + dp(1), wash);
        canvas.restoreToCount(save);

        Paint outer = paint(accent, dp(1.15f), Paint.Style.STROKE);
        outer.setAlpha(withAlpha(accent, 112) >>> 24);
        canvas.drawRoundRect(box, radius, radius, outer);
        Paint inner = paint(secondary, dp(.65f), Paint.Style.STROKE);
        inner.setAlpha(withAlpha(secondary, 48) >>> 24);
        RectF insetBox = new RectF(box.left + dp(3), box.top + dp(3), box.right - dp(3), box.bottom - dp(3));
        canvas.drawRoundRect(insetBox, dp(15), dp(15), inner);

        float corner = dp(15), margin = dp(7);
        Paint bright = paint(accent, dp(2), Paint.Style.STROKE);
        bright.setAlpha(withAlpha(accent, 220) >>> 24);
        Paint dim = paint(secondary, dp(1.3f), Paint.Style.STROKE);
        dim.setAlpha(withAlpha(secondary, 130) >>> 24);
        // Four short corner traces give the panel its projected, instrument-like edge.
        drawCorner(canvas, box.left + margin, box.top + margin, 1, 1, corner, bright);
        drawCorner(canvas, box.right - margin, box.top + margin, -1, 1, corner, dim);
        drawCorner(canvas, box.left + margin, box.bottom - margin, 1, -1, corner, dim);
        drawCorner(canvas, box.right - margin, box.bottom - margin, -1, -1, corner, bright);

        Paint tick = paint(secondary, dp(1), Paint.Style.STROKE);
        tick.setAlpha(withAlpha(secondary, 115) >>> 24);
        float x = box.right - dp(11);
        for (int i = 0; i < 3; i++) canvas.drawLine(x, box.centerY() + dp(i * 4 - 4), x - dp(3), box.centerY() + dp(i * 4 - 4), tick);
    }

    private void drawCorner(Canvas canvas, float x, float y, int dirX, int dirY, float length, Paint paint) {
        canvas.drawLine(x, y, x + dirX * length, y, paint);
        canvas.drawLine(x, y, x, y + dirY * length, paint);
    }

    @Override public void setAlpha(int alpha) { this.alpha = Math.max(0, Math.min(255, alpha)); invalidateSelf(); }
    @Override public void setColorFilter(ColorFilter colorFilter) { this.colorFilter = colorFilter; invalidateSelf(); }
    @Override public int getOpacity() { return PixelFormat.TRANSLUCENT; }
}
