// OP_Pattern pass: the init image for the diffusion node's Video Input.
// Sentinel forms x_t = sqrt(a_t) * init + sqrt(1 - a_t) * noise for every denoise below 1.0,
// so the full-contrast control image as init takes the picture over. This output pulls the
// control image toward mid grey by `init_strength` so the init only nudges composition and
// the UNet gets the near-pure-noise regime that renders sharpest around denoise 0.75.
// _Tex0 = pass:tones (the control image).
RWTexture2D<float4> OutputUAV : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint2 pixel = DTid.xy;
    if (pixel.x >= (uint)_Resolution.x || pixel.y >= (uint)_Resolution.y) return;
    float2 uv = ((float2)pixel + 0.5) / _Resolution.xy;
    float c = _Tex0.SampleLevel(LinearSampler, uv, 0).r;
    float v = lerp(0.5, c, saturate(init_strength));
    OutputUAV[pixel] = float4(v.xxx, 1.0);
}
