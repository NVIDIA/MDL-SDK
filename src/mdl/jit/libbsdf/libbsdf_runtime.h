/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************************************/

#ifndef MDL_LIBBSDF_RUNTIME_H
#define MDL_LIBBSDF_RUNTIME_H

namespace math
{
    int abs(int a);
    int2 abs(int2 const &a);
    int3 abs(int3 const &a);
    int4 abs(int4 const &a);
    float abs(float a);
    float2 abs(float2 const &a);
    float3 abs(float3 const &a);
    float4 abs(float4 const &a);
    double abs(double a);
    double2 abs(double2 const &a);
    double3 abs(double3 const &a);
    double4 abs(double4 const &a);
    color abs(color const &a);
    float acos(float a);
    float2 acos(float2 const &a);
    float3 acos(float3 const &a);
    float4 acos(float4 const &a);
    double acos(double a);
    double2 acos(double2 const &a);
    double3 acos(double3 const &a);
    double4 acos(double4 const &a);
    bool all(bool a);
    bool all(bool2 const &a);
    bool all(bool3 const &a);
    bool all(bool4 const &a);
    bool any(bool a);
    bool any(bool2 const &a);
    bool any(bool3 const &a);
    bool any(bool4 const &a);
    float asin(float a);
    float2 asin(float2 const &a);
    float3 asin(float3 const &a);
    float4 asin(float4 const &a);
    double asin(double a);
    double2 asin(double2 const &a);
    double3 asin(double3 const &a);
    double4 asin(double4 const &a);
    float atan(float a);
    float2 atan(float2 const &a);
    float3 atan(float3 const &a);
    float4 atan(float4 const &a);
    double atan(double a);
    double2 atan(double2 const &a);
    double3 atan(double3 const &a);
    double4 atan(double4 const &a);
    float atan2(float y, float x);
    float2 atan2(float2 const &y, float2 const &x);
    float3 atan2(float3 const &y, float3 const &x);
    float4 atan2(float4 const &y, float4 const &x);
    double atan2(double y, double x);
    double2 atan2(double2 const &y, double2 const &x);
    double3 atan2(double3 const &y, double3 const &x);
    double4 atan2(double4 const &y, double4 const &x);
    float average(float a);
    float average(float2 const &a);
    float average(float3 const &a);
    float average(float4 const &a);
    double average(double a);
    double average(double2 const &a);
    double average(double3 const &a);
    double average(double4 const &a);
    float average(color const &a);
    float ceil(float a);
    float2 ceil(float2 const &a);
    float3 ceil(float3 const &a);
    float4 ceil(float4 const &a);
    double ceil(double a);
    double2 ceil(double2 const &a);
    double3 ceil(double3 const &a);
    double4 ceil(double4 const &a);
    int clamp(int a, int min, int max);
    int2 clamp(int2 const &a, int2 const &min, int2 const &max);
    int3 clamp(int3 const &a, int3 const &min, int3 const &max);
    int4 clamp(int4 const &a, int4 const &min, int4 const &max);
    float clamp(float a, float min, float max);
    float2 clamp(float2 const &a, float2 const &min, float2 const &max);
    float3 clamp(float3 const &a, float3 const &min, float3 const &max);
    float4 clamp(float4 const &a, float4 const &min, float4 const &max);
    double clamp(double a, double min, double max);
    double2 clamp(double2 const &a, double2 const &min, double2 const &max);
    double3 clamp(double3 const &a, double3 const &min, double3 const &max);
    double4 clamp(double4 const &a, double4 const &min, double4 const &max);
    color clamp(color const &a, color const &min, color const &max);
    int2 clamp(int2 const &a, int2 const &min, int max);
    int2 clamp(int2 const &a, int min, int2 const &max);
    int2 clamp(int2 const &a, int min, int max);
    int3 clamp(int3 const &a, int3 const &min, int max);
    int3 clamp(int3 const &a, int min, int3 const &max);
    int3 clamp(int3 const &a, int min, int max);
    int4 clamp(int4 const &a, int4 const &min, int max);
    int4 clamp(int4 const &a, int min, int4 const &max);
    int4 clamp(int4 const &a, int min, int max);
    float2 clamp(float2 const &a, float2 const &min, float max);
    float2 clamp(float2 const &a, float min, float2 const &max);
    float2 clamp(float2 const &a, float min, float max);
    float3 clamp(float3 const &a, float3 const &min, float max);
    float3 clamp(float3 const &a, float min, float3 const &max);
    float3 clamp(float3 const &a, float min, float max);
    float4 clamp(float4 const &a, float4 const &min, float max);
    float4 clamp(float4 const &a, float min, float4 const &max);
    float4 clamp(float4 const &a, float min, float max);
    color clamp(color const &a, color const &min, float max);
    color clamp(color const &a, float min, color const &max);
    color clamp(color const &a, float min, float max);
    double2 clamp(double2 const &a, double2 const &min, double max);
    double2 clamp(double2 const &a, double min, double2 const &max);
    double2 clamp(double2 const &a, double min, double max);
    double3 clamp(double3 const &a, double3 const &min, double max);
    double3 clamp(double3 const &a, double min, double3 const &max);
    double3 clamp(double3 const &a, double min, double max);
    double4 clamp(double4 const &a, double4 const &min, double max);
    double4 clamp(double4 const &a, double min, double4 const &max);
    double4 clamp(double4 const &a, double min, double max);
    float cos(float a);
    float2 cos(float2 const &a);
    float3 cos(float3 const &a);
    float4 cos(float4 const &a);
    double cos(double a);
    double2 cos(double2 const &a);
    double3 cos(double3 const &a);
    double4 cos(double4 const &a);
    float3 cross(float3 const &a, float3 const &b);
    double3 cross(double3 const &a, double3 const &b);
    float degrees(float a);
    float2 degrees(float2 const &a);
    float3 degrees(float3 const &a);
    float4 degrees(float4 const &a);
    double degrees(double a);
    double2 degrees(double2 const &a);
    double3 degrees(double3 const &a);
    double4 degrees(double4 const &a);
    float distance(float a, float b);
    float distance(float2 const &a, float2 const &b);
    float distance(float3 const &a, float3 const &b);
    float distance(float4 const &a, float4 const &b);
    double distance(double a, double b);
    double distance(double2 const &a, double2 const &b);
    double distance(double3 const &a, double3 const &b);
    double distance(double4 const &a, double4 const &b);
    float dot(float a, float b);
    float dot(float2 const &a, float2 const &b);
    float dot(float3 const &a, float3 const &b);
    float dot(float4 const &a, float4 const &b);
    double dot(double a, double b);
    double dot(double2 const &a, double2 const &b);
    double dot(double3 const &a, double3 const &b);
    double dot(double4 const &a, double4 const &b);
    float exp(float a);
    float2 exp(float2 const &a);
    float3 exp(float3 const &a);
    float4 exp(float4 const &a);
    double exp(double a);
    double2 exp(double2 const &a);
    double3 exp(double3 const &a);
    double4 exp(double4 const &a);
    color exp(color const &a);
    float exp2(float a);
    float2 exp2(float2 const &a);
    float3 exp2(float3 const &a);
    float4 exp2(float4 const &a);
    double exp2(double a);
    double2 exp2(double2 const &a);
    double3 exp2(double3 const &a);
    double4 exp2(double4 const &a);
    color exp2(color const &a);
    float floor(float a);
    float2 floor(float2 const &a);
    float3 floor(float3 const &a);
    float4 floor(float4 const &a);
    double floor(double a);
    double2 floor(double2 const &a);
    double3 floor(double3 const &a);
    double4 floor(double4 const &a);
    float fmod(float a, float b);
    float2 fmod(float2 const &a, float2 const &b);
    float3 fmod(float3 const &a, float3 const &b);
    float4 fmod(float4 const &a, float4 const &b);
    double fmod(double a, double b);
    double2 fmod(double2 const &a, double2 const &b);
    double3 fmod(double3 const &a, double3 const &b);
    double4 fmod(double4 const &a, double4 const &b);
    float2 fmod(float2 const &a, float b);
    float3 fmod(float3 const &a, float b);
    float4 fmod(float4 const &a, float b);
    double2 fmod(double2 const &a, double b);
    double3 fmod(double3 const &a, double b);
    double4 fmod(double4 const &a, double b);
    float frac(float a);
    float2 frac(float2 const &a);
    float3 frac(float3 const &a);
    float4 frac(float4 const &a);
    double frac(double a);
    double2 frac(double2 const &a);
    double3 frac(double3 const &a);
    double4 frac(double4 const &a);
    bool isnan(float a);
    bool2 isnan(float2 const &a);
    bool3 isnan(float3 const &a);
    bool4 isnan(float4 const &a);
    bool isnan(double a);
    bool2 isnan(double2 const &a);
    bool3 isnan(double3 const &a);
    bool4 isnan(double4 const &a);
    bool isfinite(float a);
    bool2 isfinite(float2 const &a);
    bool3 isfinite(float3 const &a);
    bool4 isfinite(float4 const &a);
    bool isfinite(double a);
    bool2 isfinite(double2 const &a);
    bool3 isfinite(double3 const &a);
    bool4 isfinite(double4 const &a);
    float length(float a);
    float length(float2 const &a);
    float length(float3 const &a);
    float length(float4 const &a);
    double length(double a);
    double length(double2 const &a);
    double length(double3 const &a);
    double length(double4 const &a);
    float lerp(float a, float b, float l);
    float2 lerp(float2 const &a, float2 const &b, float2 const &l);
    float3 lerp(float3 const &a, float3 const &b, float3 const &l);
    float4 lerp(float4 const &a, float4 const &b, float4 const &l);
    double lerp(double a, double b, double l);
    double2 lerp(double2 const &a, double2 const &b, double2 const &l);
    double3 lerp(double3 const &a, double3 const &b, double3 const &l);
    double4 lerp(double4 const &a, double4 const &b, double4 const &l);
    float2 lerp(float2 const &a, float2 const &b, float l);
    float3 lerp(float3 const &a, float3 const &b, float l);
    float4 lerp(float4 const &a, float4 const &b, float l);
    double2 lerp(double2 const &a, double2 const &b, double l);
    double3 lerp(double3 const &a, double3 const &b, double l);
    double4 lerp(double4 const &a, double4 const &b, double l);
    color lerp(color const &a, color const &b, color const &l);
    color lerp(color const &a, color const &b, float l);
    float log(float a);
    float2 log(float2 const &a);
    float3 log(float3 const &a);
    float4 log(float4 const &a);
    double log(double a);
    double2 log(double2 const &a);
    double3 log(double3 const &a);
    double4 log(double4 const &a);
    color log(color const &a);
    float log2(float a);
    float2 log2(float2 const &a);
    float3 log2(float3 const &a);
    float4 log2(float4 const &a);
    double log2(double a);
    double2 log2(double2 const &a);
    double3 log2(double3 const &a);
    double4 log2(double4 const &a);
    color log2(color const &a);
    float log10(float a);
    float2 log10(float2 const &a);
    float3 log10(float3 const &a);
    float4 log10(float4 const &a);
    double log10(double a);
    double2 log10(double2 const &a);
    double3 log10(double3 const &a);
    double4 log10(double4 const &a);
    color log10(color const &a);
    float luminance(float3 const &a);
    float luminance(color const &a);
    int max(int a, int b);
    int2 max(int2 const &a, int2 const &b);
    int3 max(int3 const &a, int3 const &b);
    int4 max(int4 const &a, int4 const &b);
    float max(float a, float b);
    float2 max(float2 const &a, float2 const &b);
    float3 max(float3 const &a, float3 const &b);
    float4 max(float4 const &a, float4 const &b);
    double max(double a, double b);
    double2 max(double2 const &a, double2 const &b);
    double3 max(double3 const &a, double3 const &b);
    double4 max(double4 const &a, double4 const &b);
    color max(color const &a, color const &b);
    color max(float a, color const &b);
    color max(color const &a, float b);
    float max_value(float a);
    float max_value(float2 const &a);
    float max_value(float3 const &a);
    float max_value(float4 const &a);
    double max_value(double a);
    double max_value(double2 const &a);
    double max_value(double3 const &a);
    double max_value(double4 const &a);
    float max_value(color const &a);
    int min(int a, int b);
    int2 min(int2 const &a, int2 const &b);
    int3 min(int3 const &a, int3 const &b);
    int4 min(int4 const &a, int4 const &b);
    float min(float a, float b);
    float2 min(float2 const &a, float2 const &b);
    float3 min(float3 const &a, float3 const &b);
    float4 min(float4 const &a, float4 const &b);
    double min(double a, double b);
    double2 min(double2 const &a, double2 const &b);
    double3 min(double3 const &a, double3 const &b);
    double4 min(double4 const &a, double4 const &b);
    color min(color const &a, color const &b);
    color min(float a, color const &b);
    color min(color const &a, float b);
    float min_value(float a);
    float min_value(float2 const &a);
    float min_value(float3 const &a);
    float min_value(float4 const &a);
    double min_value(double a);
    double min_value(double2 const &a);
    double min_value(double3 const &a);
    double min_value(double4 const &a);
    float min_value(color const &a);
    void modf(float a, float* res_0, float* res_1);
    void modf(float2 const &a, float2* res_0, float2* res_1);
    void modf(float3 const &a, float3* res_0, float3* res_1);
    void modf(float4 const &a, float4* res_0, float4* res_1);
    void modf(double a, double* res_0, double* res_1);
    void modf(double2 const &a, double2* res_0, double2* res_1);
    void modf(double3 const &a, double3* res_0, double3* res_1);
    void modf(double4 const &a, double4* res_0, double4* res_1);
    float normalize(float a);
    float2 normalize(float2 const &a);
    float3 normalize(float3 const &a);
    float4 normalize(float4 const &a);
    double normalize(double a);
    double2 normalize(double2 const &a);
    double3 normalize(double3 const &a);
    double4 normalize(double4 const &a);
    int pow(int a, int b);
    int2 pow(int2 const &a, int2 const &b);
    int3 pow(int3 const &a, int3 const &b);
    int4 pow(int4 const &a, int4 const &b);
    float pow(float a, float b);
    float2 pow(float2 const &a, float2 const &b);
    float3 pow(float3 const &a, float3 const &b);
    float4 pow(float4 const &a, float4 const &b);
    double pow(double a, double b);
    double2 pow(double2 const &a, double2 const &b);
    double3 pow(double3 const &a, double3 const &b);
    double4 pow(double4 const &a, double4 const &b);
    int2 pow(int2 const &a, int b);
    int3 pow(int3 const &a, int b);
    int4 pow(int4 const &a, int b);
    float2 pow(float2 const &a, float b);
    float3 pow(float3 const &a, float b);
    float4 pow(float4 const &a, float b);
    double2 pow(double2 const &a, double b);
    double3 pow(double3 const &a, double b);
    double4 pow(double4 const &a, double b);
    color pow(color const &a, color const &b);
    color pow(color const &a, float b);
    float radians(float a);
    float2 radians(float2 const &a);
    float3 radians(float3 const &a);
    float4 radians(float4 const &a);
    double radians(double a);
    double2 radians(double2 const &a);
    double3 radians(double3 const &a);
    double4 radians(double4 const &a);
    float round(float a);
    float2 round(float2 const &a);
    float3 round(float3 const &a);
    float4 round(float4 const &a);
    double round(double a);
    double2 round(double2 const &a);
    double3 round(double3 const &a);
    double4 round(double4 const &a);
    float rsqrt(float a);
    float2 rsqrt(float2 const &a);
    float3 rsqrt(float3 const &a);
    float4 rsqrt(float4 const &a);
    double rsqrt(double a);
    double2 rsqrt(double2 const &a);
    double3 rsqrt(double3 const &a);
    double4 rsqrt(double4 const &a);
    color rsqrt(color const &a);
    float saturate(float a);
    float2 saturate(float2 const &a);
    float3 saturate(float3 const &a);
    float4 saturate(float4 const &a);
    double saturate(double a);
    double2 saturate(double2 const &a);
    double3 saturate(double3 const &a);
    double4 saturate(double4 const &a);
    color saturate(color const &a);
    int sign(int a);
    int2 sign(int2 const &a);
    int3 sign(int3 const &a);
    int4 sign(int4 const &a);
    float sign(float a);
    float2 sign(float2 const &a);
    float3 sign(float3 const &a);
    float4 sign(float4 const &a);
    double sign(double a);
    double2 sign(double2 const &a);
    double3 sign(double3 const &a);
    double4 sign(double4 const &a);
    float sin(float a);
    float2 sin(float2 const &a);
    float3 sin(float3 const &a);
    float4 sin(float4 const &a);
    double sin(double a);
    double2 sin(double2 const &a);
    double3 sin(double3 const &a);
    double4 sin(double4 const &a);
    void sincos(float a, float* res_0, float* res_1);
    void sincos(float2 const &a, float2* res_0, float2* res_1);
    void sincos(float3 const &a, float3* res_0, float3* res_1);
    void sincos(float4 const &a, float4* res_0, float4* res_1);
    void sincos(double a, double* res_0, double* res_1);
    void sincos(double2 const &a, double2* res_0, double2* res_1);
    void sincos(double3 const &a, double3* res_0, double3* res_1);
    void sincos(double4 const &a, double4* res_0, double4* res_1);
    float smoothstep(float a, float b, float l);
    float2 smoothstep(float2 const &a, float2 const &b, float2 const &l);
    float3 smoothstep(float3 const &a, float3 const &b, float3 const &l);
    float4 smoothstep(float4 const &a, float4 const &b, float4 const &l);
    double smoothstep(double a, double b, double l);
    double2 smoothstep(double2 const &a, double2 const &b, double2 const &l);
    double3 smoothstep(double3 const &a, double3 const &b, double3 const &l);
    double4 smoothstep(double4 const &a, double4 const &b, double4 const &l);
    float2 smoothstep(float2 const &a, float2 const &b, float l);
    float3 smoothstep(float3 const &a, float3 const &b, float l);
    float4 smoothstep(float4 const &a, float4 const &b, float l);
    double2 smoothstep(double2 const &a, double2 const &b, double l);
    double3 smoothstep(double3 const &a, double3 const &b, double l);
    double4 smoothstep(double4 const &a, double4 const &b, double l);
    float sqrt(float a);
    float2 sqrt(float2 const &a);
    float3 sqrt(float3 const &a);
    float4 sqrt(float4 const &a);
    double sqrt(double a);
    double2 sqrt(double2 const &a);
    double3 sqrt(double3 const &a);
    double4 sqrt(double4 const &a);
    color sqrt(color const &a);
    float step(float a, float b);
    float2 step(float2 const &a, float2 const &b);
    float3 step(float3 const &a, float3 const &b);
    float4 step(float4 const &a, float4 const &b);
    double step(double a, double b);
    double2 step(double2 const &a, double2 const &b);
    double3 step(double3 const &a, double3 const &b);
    double4 step(double4 const &a, double4 const &b);
    float tan(float a);
    float2 tan(float2 const &a);
    float3 tan(float3 const &a);
    float4 tan(float4 const &a);
    double tan(double a);
    double2 tan(double2 const &a);
    double3 tan(double3 const &a);
    double4 tan(double4 const &a);
    // float2x2 transpose(float2x2 a);  (not supported yet)
    // float2x3 transpose(float3x2 a);  (not supported yet)
    // float3x2 transpose(float2x3 a);  (not supported yet)
    // float3x3 transpose(float3x3 a);  (not supported yet)
    // float4x2 transpose(float2x4 a);  (not supported yet)
    // float2x4 transpose(float4x2 a);  (not supported yet)
    // float3x4 transpose(float4x3 a);  (not supported yet)
    // float4x3 transpose(float3x4 a);  (not supported yet)
    // float4x4 transpose(float4x4 a);  (not supported yet)
    // double2x2 transpose(double2x2 a);  (not supported yet)
    // double2x3 transpose(double3x2 a);  (not supported yet)
    // double3x2 transpose(double2x3 a);  (not supported yet)
    // double3x3 transpose(double3x3 a);  (not supported yet)
    // double4x2 transpose(double2x4 a);  (not supported yet)
    // double2x4 transpose(double4x2 a);  (not supported yet)
    // double3x4 transpose(double4x3 a);  (not supported yet)
    // double4x3 transpose(double3x4 a);  (not supported yet)
    // double4x4 transpose(double4x4 a);  (not supported yet)
    color blackbody(float temperature);
    // color emission_color(float[<N>] wavelengths, float[N] amplitudes);  (not supported yet)
    color emission_color(color const &value);
}

namespace debug
{
    bool breakpoint();
    bool assert(bool condition, string reason, string funcname = "", string filename = "",
        int line = 0);
    bool print(bool v);
    bool print(bool2 const &v);
    bool print(bool3 const &v);
    bool print(bool4 const &v);
    bool print(int v);
    bool print(int2 const &v);
    bool print(int3 const &v);
    bool print(int4 const &v);
    bool print(float v);
    bool print(float2 const &v);
    bool print(float3 const &v);
    bool print(float4 const &v);
    bool print(double v);
    bool print(double2 const &v);
    bool print(double3 const &v);
    bool print(double4 const &v);
    bool print(color const &v);
    bool print(string v);
}

#endif  // MDL_LIBBSDF_RUNTIME_H
