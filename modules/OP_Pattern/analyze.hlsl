// OP_Pattern analysis pass (single thread group, one buffer writer).
// Builds a 64-bin luminance histogram of the source on a 224x128 sample grid in groupshared
// memory, then thread 0 derives the two percentile thresholds that split the frame into
// darkest `coverage` / mid / brightest `coverage`, smooths them, and integrates pattern phase.
//
// _Tex0 = source video (auto-declared). Output UAV = stat buffer.
// Stat layout (uint; floats stored via asuint):
//   [0..63] histogram bins        [64] low thr (smoothed)   [65] high thr (smoothed)
//   [66] phase (cycles)           [67] measured dark cov     [68] measured light cov
//   [69] total samples            [70] low raw              [71] high raw
//   [72] frame counter            [73] last _Time
RWStructuredBuffer<uint> Stat : register(u0);

static const uint2 GRID = uint2(224u, 128u);
static const uint THREADS = 256u;
static const uint SAMPLES_PER_THREAD = (224u * 128u) / THREADS;   // 112

groupshared uint gBins[64];

float Luma(float3 c) { return dot(c, float3(0.299, 0.587, 0.114)); }

float PercentileFromBottom(float target) {
    float cum = 0.0;
    [loop]
    for (uint b = 0u; b < 64u; ++b) {
        float n = (float)gBins[b];
        if (cum + n >= target) {
            float f = (n > 0.0) ? saturate((target - cum) / n) : 0.0;
            return ((float)b + f) / 64.0;
        }
        cum += n;
    }
    return 1.0;
}

float PercentileFromTop(float target) {
    float cum = 0.0;
    [loop]
    for (int b = 63; b >= 0; --b) {
        float n = (float)gBins[b];
        if (cum + n >= target) {
            float f = (n > 0.0) ? saturate((target - cum) / n) : 0.0;
            return ((float)b + 1.0 - f) / 64.0;
        }
        cum += n;
    }
    return 0.0;
}

float CountBelow(float thr) {
    float cum = 0.0;
    float edge = thr * 64.0;
    [loop]
    for (uint b = 0u; b < 64u; ++b) cum += (float)gBins[b] * saturate(edge - (float)b);
    return cum;
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_GroupThreadID) {
    uint t = tid.x;
    if (t < 64u) gBins[t] = 0u;
    GroupMemoryBarrierWithGroupSync();

    // Each thread walks a strided subset of the sample grid.
    [loop]
    for (uint i = 0u; i < SAMPLES_PER_THREAD; ++i) {
        uint s = t + i * THREADS;
        uint2 g = uint2(s % GRID.x, s / GRID.x);
        float2 uv = ((float2)g + 0.5) / (float2)GRID;
        float l = saturate(Luma(_Tex0.SampleLevel(LinearSampler, uv, 0).rgb));
        uint bin = min((uint)(l * 64.0), 63u);
        InterlockedAdd(gBins[bin], 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    if (t < 64u) Stat[t] = gBins[t];
    if (t != 0u) return;

    // Timing from the stored last time so smoothing is frame-rate independent.
    uint frame = Stat[72];
    float last = asfloat(Stat[73]);
    float dt = (frame == 0u) ? 0.0 : clamp(_Time - last, 0.0, 0.1);
    Stat[73] = asuint(_Time);

    float total = 0.0;
    [loop]
    for (uint b = 0u; b < 64u; ++b) total += (float)gBins[b];
    Stat[69] = asuint(total);

    float cov = clamp(coverage, 0.01, 0.49);
    float target = total * cov;
    float lowRaw = (total > 0.0) ? PercentileFromBottom(target) : 0.25;
    float highRaw = (total > 0.0) ? PercentileFromTop(target) : 0.75;
    highRaw = max(highRaw, lowRaw + 1.0 / 64.0);
    Stat[70] = asuint(lowRaw);
    Stat[71] = asuint(highRaw);

    float k = (frame == 0u) ? 1.0 : (1.0 - exp(-dt * max(smoothing, 0.01)));
    float low = lerp(asfloat(Stat[64]), lowRaw, k);
    float high = lerp(asfloat(Stat[65]), highRaw, k);
    Stat[64] = asuint(low);
    Stat[65] = asuint(high);

    // Measured coverage at the smoothed thresholds (what the tones pass actually applies).
    Stat[67] = asuint((total > 0.0) ? CountBelow(low) / total : 0.0);
    Stat[68] = asuint((total > 0.0) ? 1.0 - CountBelow(high) / total : 0.0);

    // Pattern phase: rate-class, so integrate rather than multiply absolute time.
    float phase = asfloat(Stat[66]) + spin * dt;
    phase -= floor(phase);
    Stat[66] = asuint(phase);

    Stat[72] = frame + 1u;
}
