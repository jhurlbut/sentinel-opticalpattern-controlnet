// OP_Pattern pass 4: the control image. Three tones only: black where the ControlNet should
// see "dark", white where it should see "light", mid grey everywhere it should stay free.
// _Tex0 = source video (auto-declared at t0). Stat buffer at t1.
RWTexture2D<float4> OutputUAV : register(u0);
StructuredBuffer<uint> Stat : register(t1);

static const float PI = 3.14159265;

float Luma(float3 c) { return dot(c, float3(0.299, 0.587, 0.114)); }

// Map a scalar v in [0,1) onto the three tones with `cov` at each end.
float ThreeTone(float v, float cov, float s) {
    float dark = 1.0 - smoothstep(cov - s, cov + s, v);
    float light = smoothstep(1.0 - cov - s, 1.0 - cov + s, v);
    float value = lerp(0.5, 0.0, dark);
    return lerp(value, 1.0, light);
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint2 pixel = DTid.xy;
    if (pixel.x >= (uint)_Resolution.x || pixel.y >= (uint)_Resolution.y) return;
    float2 uv = ((float2)pixel + 0.5) / _Resolution.xy;

    float low = asfloat(Stat[64]);
    float high = asfloat(Stat[65]);
    float phase = asfloat(Stat[66]);
    float cov = clamp(coverage, 0.01, 0.49);
    float s = max(softness, 0.0005);

    float value = 0.5;
    if (pattern == 0) {
        // Source tones: percentile split of the live source luminance.
        float l = Luma(_Tex0.SampleLevel(LinearSampler, uv, 0).rgb);
        float dark = 1.0 - smoothstep(low - s, low + s, l);
        float light = smoothstep(high - s, high + s, l);
        value = lerp(0.5, 0.0, dark);
        value = lerp(value, 1.0, light);
    } else {
        float aspect = _Resolution.x / _Resolution.y;
        float2 p = (uv - 0.5) * float2(aspect, 1.0) * scale;
        if (pattern == 1) {
            // Spiral: arms wound by log-radius twist.
            float r = length(p);
            float a = atan2(p.y, p.x) / (2.0 * PI);
            float v = frac(a * (float)arms + log(max(r, 0.002)) * twist - phase);
            value = ThreeTone(v, cov, s);
        } else if (pattern == 2) {
            // Rings.
            float v = frac(length(p) * 2.0 - phase);
            value = ThreeTone(v, cov, s);
        } else if (pattern == 3) {
            // Stripes at `angle`.
            float ang = radians(angle);
            float d = p.x * cos(ang) + p.y * sin(ang);
            value = ThreeTone(frac(d - phase), cov, s);
        } else {
            // Checker: alternating black/white cells with a grey margin sized so each tone
            // still covers `cov` of the frame: 0.5 * (1 - 2m)^2 = cov.
            float2 q = p + phase;
            float2 cell = floor(q);
            float2 f = frac(q);
            float parity = fmod(abs(cell.x + cell.y), 2.0);
            float e = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y));
            float m = 0.5 * (1.0 - sqrt(saturate(2.0 * cov)));
            float inside = smoothstep(m - s, m + s, e);
            value = lerp(0.5, parity > 0.5 ? 1.0 : 0.0, inside);
        }
    }

    if (invert != 0) value = 1.0 - value;
    OutputUAV[pixel] = float4(value.xxx, 1.0);
}
