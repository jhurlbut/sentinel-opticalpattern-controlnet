// OP_Pattern pass 5: instrument view. Control image inset top-left, coverage bars on the
// right, the live luminance histogram with both thresholds along the bottom.
// _Tex0 = pass:tones (auto-declared at t0). Stat buffer at t1.
#include "../_shared/plan_theme.hlsli"

RWTexture2D<float4> OutputUAV : register(u0);
StructuredBuffer<uint> Stat : register(t1);

static const float2 DESIGN = float2(896.0, 512.0);
static const float IMG_W = 672.0;   // control inset width  (896 * 0.75)
static const float IMG_H = 384.0;   // control inset height (512 * 0.75)
static const float HIST_Y0 = 400.0; // histogram strip top
static const float HIST_Y1 = 500.0; // histogram strip bottom
static const float HIST_X0 = 16.0;
static const float HIST_X1 = 656.0; // 640 px for 64 bins = 10 px each

float Bar(float x, float y, float2 lo, float2 hi) {
    return (x >= lo.x && x < hi.x && y >= lo.y && y < hi.y) ? 1.0 : 0.0;
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint2 pixel = DTid.xy;
    if (pixel.x >= (uint)_Resolution.x || pixel.y >= (uint)_Resolution.y) return;
    float2 d = ((float2)pixel + 0.5) * DESIGN / max(_Resolution.xy, float2(1.0, 1.0));

    float low = asfloat(Stat[64]);
    float high = asfloat(Stat[65]);
    float covDark = asfloat(Stat[67]);
    float covLight = asfloat(Stat[68]);
    float total = max(asfloat(Stat[69]), 1.0);
    float cov = clamp(coverage, 0.01, 0.49);

    float3 col = PT_FIELD;

    // --- Control inset (data, not chrome) ---
    if (d.x < IMG_W && d.y < IMG_H) {
        float2 iuv = d / float2(IMG_W, IMG_H);
        float3 c = _Tex0.SampleLevel(LinearSampler, iuv, 0).rgb;
        col = ptInset(c);
    }
    // inset frame
    if ((abs(d.x - IMG_W) < 1.0 && d.y < IMG_H) || (abs(d.y - IMG_H) < 1.0 && d.x <= IMG_W)) col = PT_RULE;

    // --- Coverage bars, right column: target (grey rule) vs measured (accent) ---
    // dark bar x in [704,768), light bar x in [800,864), y from 32 (top=0.5) to 352 (0.0)
    float barTop = 32.0, barBot = 352.0;
    float2 lanes[2] = { float2(704.0, 768.0), float2(800.0, 864.0) };
    float measured[2] = { covDark, covLight };
    [unroll]
    for (int i = 0; i < 2; ++i) {
        float2 ln = lanes[i];
        if (d.x >= ln.x && d.x < ln.y && d.y >= barTop && d.y <= barBot) {
            col = PT_WELL;
            float yMeasured = barBot - saturate(measured[i] / 0.5) * (barBot - barTop);
            float yTarget = barBot - saturate(cov / 0.5) * (barBot - barTop);
            if (d.y >= yMeasured) col = (i == 0) ? PT_DIM : PT_MID;      // filled bar, tone by lane
            if (abs(d.y - yTarget) < 1.0) col = PT_INK;                  // target line
            if (abs(d.y - yMeasured) < 1.5) col = PT_ACCENT;             // live reading
        }
        // lane swatch under each bar: black for dark lane, white for light lane
        if (d.x >= ln.x && d.x < ln.y && d.y > barBot + 8.0 && d.y < barBot + 24.0)
            col = (i == 0) ? float3(0.0, 0.0, 0.0) : PT_INK;
    }

    // --- Histogram strip ---
    if (d.y >= HIST_Y0 && d.y <= HIST_Y1 && d.x >= HIST_X0 && d.x <= HIST_X1) {
        col = PT_WELL;
        float binW = (HIST_X1 - HIST_X0) / 64.0;
        uint b = min((uint)((d.x - HIST_X0) / binW), 63u);
        // normalise by the largest bin
        float mx = 1.0;
        [loop]
        for (uint k = 0u; k < 64u; ++k) mx = max(mx, (float)Stat[k]);
        float h = (float)Stat[b] / mx;
        float yTop = HIST_Y1 - h * (HIST_Y1 - HIST_Y0 - 4.0);
        float lumaHere = (d.x - HIST_X0) / (HIST_X1 - HIST_X0);
        // Region tone: bins below low read dark, above high read light, middle mid.
        float3 barCol = PT_GRID;
        if (lumaHere < low) barCol = PT_DIM;
        else if (lumaHere > high) barCol = PT_MID;
        if (d.y >= yTop) col = barCol;
        // threshold markers: the live reading, accent
        float xLow = HIST_X0 + low * (HIST_X1 - HIST_X0);
        float xHigh = HIST_X0 + high * (HIST_X1 - HIST_X0);
        if (abs(d.x - xLow) < 1.0 || abs(d.x - xHigh) < 1.0) col = PT_ACCENT;
    }
    // grey ramp under the histogram: what luma each column means
    if (d.y > HIST_Y1 + 2.0 && d.y < HIST_Y1 + 8.0 && d.x >= HIST_X0 && d.x <= HIST_X1) {
        float l = (d.x - HIST_X0) / (HIST_X1 - HIST_X0);
        col = l.xxx * 0.9;
    }

    // Source-tones mode is the only one where the histogram drives the output; in pattern
    // modes dim the strip so it reads as context, not as the live reading.
    if (pattern != 0 && d.y >= HIST_Y0) col = lerp(PT_FIELD, col, 0.45);

    OutputUAV[pixel] = float4(col, 1.0);
}
