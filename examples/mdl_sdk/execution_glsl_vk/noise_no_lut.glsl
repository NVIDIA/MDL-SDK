/******************************************************************************
 * Copyright 2023 NVIDIA Corporation. All rights reserved.
 *****************************************************************************/

const float PI = 3.14159265359;
const float ONE_OVER_PI = 0.3183099;

int hash(int seed, int i)
{
    return (i ^ seed) * 1075385539;
}

uint rnd_init(int px, int py, int pz)
{
    return uint(hash(hash(hash(0, px), py), pz));
}

uint rnd_next(uint state) {
    // xorshift32
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
float rnd_value(uint state) {

    return float(state) * 2.3283064365386963e-10;
}

float glattice(int px, int py, int pz, float fx, float fy, float fz)
{
    uint seed = rnd_init(px, py, pz);
    
    seed = rnd_next(seed);
    float xi0 = rnd_value(seed);
    seed = rnd_next(seed);
    float xi1 = rnd_value(seed);
    
    vec3 grad;
    grad.z = 1.0f - 2.0f * xi1;
    float sintheta = sqrt(1.0f - grad.z * grad.z);
    float phi = 2.0f * (PI * xi0);
    grad.x = cos(phi) * sintheta;
    grad.y = sin(phi) * sintheta;

    return grad.x * fx + grad.y * fy + grad.z * fz;
}

// TODO: No need to overload?!
/*
float mix(float x, float y, float t)
{
    return x + t * (y - x);
}
*/

float smootherstep(float t)
{
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

float noise_float3(vec3 pos)
{
    vec3 base_pos = floor(pos);

    float fx0 = pos.x - base_pos.x;
    float fy0 = pos.y - base_pos.y;
    float fz0 = pos.z - base_pos.z;
    float fx1 = fx0 - 1.0f;
    float fy1 = fy0 - 1.0f;
    float fz1 = fz0 - 1.0f;
    
    float wx = smootherstep(fx0);
    float wy = smootherstep(fy0);
    float wz = smootherstep(fz0);

    int ix0 = int(base_pos.x);
    int iy0 = int(base_pos.y);
    int iz0 = int(base_pos.z);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;
    int iz1 = iz0 + 1;

    return mix(
        mix(
            mix(glattice(ix0, iy0, iz0, fx0, fy0, fz0), glattice(ix1, iy0, iz0, fx1, fy0, fz0), wx),
            mix(glattice(ix0, iy1, iz0, fx0, fy1, fz0), glattice(ix1, iy1, iz0, fx1, fy1, fz0), wx),
            wy),
        mix(
            mix(glattice(ix0, iy0, iz1, fx0, fy0, fz1), glattice(ix1, iy0, iz1, fx1, fy0, fz1), wx),
            mix(glattice(ix0, iy1, iz1, fx0, fy1, fz1), glattice(ix1, iy1, iz1, fx1, fy1, fz1), wx),
            wy),
        wz);
}

#ifdef MAPPED__ZN4base12perlin_noiseEu6float4
float noise_float4(vec4 v)
{
    // todo: implement
    return noise_float3(vec3(v.xyz));
}
#endif

#ifdef MAPPED__ZN4base12worley_noiseEu6float3fi
_ZN4base13worley_returnE noise_worley(vec3 pos, float jitter, int metric)
{
    _ZN4base13worley_returnE ret = _ZN4base13worley_returnE(vec3(0.0), vec3(0.0), vec2(0.0));
    vec3 cell = vec3(floor(pos.x), floor(pos.y), floor(pos.z));
    vec2 f1f2 = vec2(3.402823e+38, 3.402823e+38);
   
    for (int i = -1; i <= 1; ++i) {
        float localcellx = cell.x + float(i);
        int X = int(floor(localcellx));
        for (int j = -1; j <= 1; ++j) {
            float localcelly = cell.y + float(j);
            int Y = int(floor(localcelly));
            for (int k = -1; k <= 1; ++k) {
                
                float localcellz = cell.z + float(k);
                int Z = int(floor(localcellz));
                
                uint seed = rnd_init(X, Y, Z);
                seed = rnd_next(seed);
                float offset_x = rnd_value(seed);
                seed = rnd_next(seed);
                float offset_y = rnd_value(seed);
                seed = rnd_next(seed);
                float offset_z = rnd_value(seed);
                
                vec3 localpos = vec3(localcellx, localcelly, localcellz) + 
                    vec3(offset_x, offset_y, offset_z) * jitter;
                vec3 diff = localpos - pos;
                float dist = dot(diff, diff);
                if (dist < f1f2.x) {
                    f1f2.y = f1f2.x;
                    ret.nearest_pos_1 = ret.nearest_pos_0;
                    f1f2.x = dist;
                    ret.nearest_pos_0 = localpos;
                } else if (dist < f1f2.y) {
                    f1f2.y = dist;
                    ret.nearest_pos_1 = localpos;
                }
            }
        }
    }
    ret.val = metric == 0 ? vec2(sqrt(f1f2.x), sqrt(f1f2.y)) : f1f2;
    return ret;
}
#endif

#ifdef MAPPED__ZN4base8mi_noiseEu6float3
_ZN4base12noise_returnE noise_mi_float3(vec3 xyz)
{
    float val = noise_float3(xyz)* 0.5 + 0.5;
    vec3 grad = vec3(noise_float3(xyz.yzx), noise_float3(xyz.zxy), noise_float3(xyz.zyx));
    grad = grad * 0.5 + vec3(0.5);
    return _ZN4base12noise_returnE(grad, val);
}
#endif

#ifdef MAPPED__ZN4base8mi_noiseEu4int3
_ZN4base12noise_returnE noise_mi_int3(ivec3 xyz)
{
    return noise_mi_float3(vec3(xyz) + vec3(0.5));
}
#endif
