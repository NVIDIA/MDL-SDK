/***************************************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_USER_MODULES_MDL_RUNTIME_H
#define MDL_USER_MODULES_MDL_RUNTIME_H

namespace math
{
    int abs(int a);
    vint2 abs(vint2 a);
    vint3 abs(vint3 a);
    vint4 abs(vint4 a);
    float abs(float a);
    vfloat2 abs(vfloat2 a);
    vfloat3 abs(vfloat3 a);
    vfloat4 abs(vfloat4 a);
    double abs(double a);
    vdouble2 abs(vdouble2 a);
    vdouble3 abs(vdouble3 a);
    vdouble4 abs(vdouble4 a);
    // color abs(color const &a);  (not supported yet)
    float acos(float a);
    vfloat2 acos(vfloat2 a);
    vfloat3 acos(vfloat3 a);
    vfloat4 acos(vfloat4 a);
    double acos(double a);
    vdouble2 acos(vdouble2 a);
    vdouble3 acos(vdouble3 a);
    vdouble4 acos(vdouble4 a);
    bool all(bool a);
    bool all(vbool2 a);
    bool all(vbool3 a);
    bool all(vbool4 a);
    bool any(bool a);
    bool any(vbool2 a);
    bool any(vbool3 a);
    bool any(vbool4 a);
    float asin(float a);
    vfloat2 asin(vfloat2 a);
    vfloat3 asin(vfloat3 a);
    vfloat4 asin(vfloat4 a);
    double asin(double a);
    vdouble2 asin(vdouble2 a);
    vdouble3 asin(vdouble3 a);
    vdouble4 asin(vdouble4 a);
    float atan(float a);
    vfloat2 atan(vfloat2 a);
    vfloat3 atan(vfloat3 a);
    vfloat4 atan(vfloat4 a);
    double atan(double a);
    vdouble2 atan(vdouble2 a);
    vdouble3 atan(vdouble3 a);
    vdouble4 atan(vdouble4 a);
    float atan2(float y, float x);
    vfloat2 atan2(vfloat2 y, vfloat2 x);
    vfloat3 atan2(vfloat3 y, vfloat3 x);
    vfloat4 atan2(vfloat4 y, vfloat4 x);
    double atan2(double y, double x);
    vdouble2 atan2(vdouble2 y, vdouble2 x);
    vdouble3 atan2(vdouble3 y, vdouble3 x);
    vdouble4 atan2(vdouble4 y, vdouble4 x);
    float average(float a);
    float average(vfloat2 a);
    float average(vfloat3 a);
    float average(vfloat4 a);
    double average(double a);
    double average(vdouble2 a);
    double average(vdouble3 a);
    double average(vdouble4 a);
    // float average(color const &a);  (not supported yet)
    float ceil(float a);
    vfloat2 ceil(vfloat2 a);
    vfloat3 ceil(vfloat3 a);
    vfloat4 ceil(vfloat4 a);
    double ceil(double a);
    vdouble2 ceil(vdouble2 a);
    vdouble3 ceil(vdouble3 a);
    vdouble4 ceil(vdouble4 a);
    int clamp(int a, int min, int max);
    vint2 clamp(vint2 a, vint2 min, vint2 max);
    vint3 clamp(vint3 a, vint3 min, vint3 max);
    vint4 clamp(vint4 a, vint4 min, vint4 max);
    float clamp(float a, float min, float max);
    vfloat2 clamp(vfloat2 a, vfloat2 min, vfloat2 max);
    vfloat3 clamp(vfloat3 a, vfloat3 min, vfloat3 max);
    vfloat4 clamp(vfloat4 a, vfloat4 min, vfloat4 max);
    double clamp(double a, double min, double max);
    vdouble2 clamp(vdouble2 a, vdouble2 min, vdouble2 max);
    vdouble3 clamp(vdouble3 a, vdouble3 min, vdouble3 max);
    vdouble4 clamp(vdouble4 a, vdouble4 min, vdouble4 max);
    // color clamp(color const &a, color const &min, color const &max);  (not supported yet)
    vint2 clamp(vint2 a, vint2 min, int max);
    vint2 clamp(vint2 a, int min, vint2 max);
    vint2 clamp(vint2 a, int min, int max);
    vint3 clamp(vint3 a, vint3 min, int max);
    vint3 clamp(vint3 a, int min, vint3 max);
    vint3 clamp(vint3 a, int min, int max);
    vint4 clamp(vint4 a, vint4 min, int max);
    vint4 clamp(vint4 a, int min, vint4 max);
    vint4 clamp(vint4 a, int min, int max);
    vfloat2 clamp(vfloat2 a, vfloat2 min, float max);
    vfloat2 clamp(vfloat2 a, float min, vfloat2 max);
    vfloat2 clamp(vfloat2 a, float min, float max);
    vfloat3 clamp(vfloat3 a, vfloat3 min, float max);
    vfloat3 clamp(vfloat3 a, float min, vfloat3 max);
    vfloat3 clamp(vfloat3 a, float min, float max);
    vfloat4 clamp(vfloat4 a, vfloat4 min, float max);
    vfloat4 clamp(vfloat4 a, float min, vfloat4 max);
    vfloat4 clamp(vfloat4 a, float min, float max);
    // color clamp(color const &a, color const &min, float max);  (not supported yet)
    // color clamp(color const &a, float min, color const &max);  (not supported yet)
    // color clamp(color const &a, float min, float max);  (not supported yet)
    vdouble2 clamp(vdouble2 a, vdouble2 min, double max);
    vdouble2 clamp(vdouble2 a, double min, vdouble2 max);
    vdouble2 clamp(vdouble2 a, double min, double max);
    vdouble3 clamp(vdouble3 a, vdouble3 min, double max);
    vdouble3 clamp(vdouble3 a, double min, vdouble3 max);
    vdouble3 clamp(vdouble3 a, double min, double max);
    vdouble4 clamp(vdouble4 a, vdouble4 min, double max);
    vdouble4 clamp(vdouble4 a, double min, vdouble4 max);
    vdouble4 clamp(vdouble4 a, double min, double max);
    float cos(float a);
    vfloat2 cos(vfloat2 a);
    vfloat3 cos(vfloat3 a);
    vfloat4 cos(vfloat4 a);
    double cos(double a);
    vdouble2 cos(vdouble2 a);
    vdouble3 cos(vdouble3 a);
    vdouble4 cos(vdouble4 a);
    vfloat3 cross(vfloat3 a, vfloat3 b);
    vdouble3 cross(vdouble3 a, vdouble3 b);
    float degrees(float a);
    vfloat2 degrees(vfloat2 a);
    vfloat3 degrees(vfloat3 a);
    vfloat4 degrees(vfloat4 a);
    double degrees(double a);
    vdouble2 degrees(vdouble2 a);
    vdouble3 degrees(vdouble3 a);
    vdouble4 degrees(vdouble4 a);
    float distance(float a, float b);
    float distance(vfloat2 a, vfloat2 b);
    float distance(vfloat3 a, vfloat3 b);
    float distance(vfloat4 a, vfloat4 b);
    double distance(double a, double b);
    double distance(vdouble2 a, vdouble2 b);
    double distance(vdouble3 a, vdouble3 b);
    double distance(vdouble4 a, vdouble4 b);
    float dot(float a, float b);
    float dot(vfloat2 a, vfloat2 b);
    float dot(vfloat3 a, vfloat3 b);
    float dot(vfloat4 a, vfloat4 b);
    double dot(double a, double b);
    double dot(vdouble2 a, vdouble2 b);
    double dot(vdouble3 a, vdouble3 b);
    double dot(vdouble4 a, vdouble4 b);
    float exp(float a);
    vfloat2 exp(vfloat2 a);
    vfloat3 exp(vfloat3 a);
    vfloat4 exp(vfloat4 a);
    double exp(double a);
    vdouble2 exp(vdouble2 a);
    vdouble3 exp(vdouble3 a);
    vdouble4 exp(vdouble4 a);
    // color exp(color const &a);  (not supported yet)
    float exp2(float a);
    vfloat2 exp2(vfloat2 a);
    vfloat3 exp2(vfloat3 a);
    vfloat4 exp2(vfloat4 a);
    double exp2(double a);
    vdouble2 exp2(vdouble2 a);
    vdouble3 exp2(vdouble3 a);
    vdouble4 exp2(vdouble4 a);
    // color exp2(color const &a);  (not supported yet)
    float floor(float a);
    vfloat2 floor(vfloat2 a);
    vfloat3 floor(vfloat3 a);
    vfloat4 floor(vfloat4 a);
    double floor(double a);
    vdouble2 floor(vdouble2 a);
    vdouble3 floor(vdouble3 a);
    vdouble4 floor(vdouble4 a);
    float fmod(float a, float b);
    vfloat2 fmod(vfloat2 a, vfloat2 b);
    vfloat3 fmod(vfloat3 a, vfloat3 b);
    vfloat4 fmod(vfloat4 a, vfloat4 b);
    double fmod(double a, double b);
    vdouble2 fmod(vdouble2 a, vdouble2 b);
    vdouble3 fmod(vdouble3 a, vdouble3 b);
    vdouble4 fmod(vdouble4 a, vdouble4 b);
    vfloat2 fmod(vfloat2 a, float b);
    vfloat3 fmod(vfloat3 a, float b);
    vfloat4 fmod(vfloat4 a, float b);
    vdouble2 fmod(vdouble2 a, double b);
    vdouble3 fmod(vdouble3 a, double b);
    vdouble4 fmod(vdouble4 a, double b);
    float frac(float a);
    vfloat2 frac(vfloat2 a);
    vfloat3 frac(vfloat3 a);
    vfloat4 frac(vfloat4 a);
    double frac(double a);
    vdouble2 frac(vdouble2 a);
    vdouble3 frac(vdouble3 a);
    vdouble4 frac(vdouble4 a);
    bool isnan(float a);
    vbool2 isnan(vfloat2 a);
    vbool3 isnan(vfloat3 a);
    vbool4 isnan(vfloat4 a);
    bool isnan(double a);
    vbool2 isnan(vdouble2 a);
    vbool3 isnan(vdouble3 a);
    vbool4 isnan(vdouble4 a);
    bool isfinite(float a);
    vbool2 isfinite(vfloat2 a);
    vbool3 isfinite(vfloat3 a);
    vbool4 isfinite(vfloat4 a);
    bool isfinite(double a);
    vbool2 isfinite(vdouble2 a);
    vbool3 isfinite(vdouble3 a);
    vbool4 isfinite(vdouble4 a);
    float length(float a);
    float length(vfloat2 a);
    float length(vfloat3 a);
    float length(vfloat4 a);
    double length(double a);
    double length(vdouble2 a);
    double length(vdouble3 a);
    double length(vdouble4 a);
    float lerp(float a, float b, float l);
    vfloat2 lerp(vfloat2 a, vfloat2 b, vfloat2 l);
    vfloat3 lerp(vfloat3 a, vfloat3 b, vfloat3 l);
    vfloat4 lerp(vfloat4 a, vfloat4 b, vfloat4 l);
    double lerp(double a, double b, double l);
    vdouble2 lerp(vdouble2 a, vdouble2 b, vdouble2 l);
    vdouble3 lerp(vdouble3 a, vdouble3 b, vdouble3 l);
    vdouble4 lerp(vdouble4 a, vdouble4 b, vdouble4 l);
    vfloat2 lerp(vfloat2 a, vfloat2 b, float l);
    vfloat3 lerp(vfloat3 a, vfloat3 b, float l);
    vfloat4 lerp(vfloat4 a, vfloat4 b, float l);
    vdouble2 lerp(vdouble2 a, vdouble2 b, double l);
    vdouble3 lerp(vdouble3 a, vdouble3 b, double l);
    vdouble4 lerp(vdouble4 a, vdouble4 b, double l);
    // color lerp(color const &a, color const &b, color const &l);  (not supported yet)
    // color lerp(color const &a, color const &b, float l);  (not supported yet)
    float log(float a);
    vfloat2 log(vfloat2 a);
    vfloat3 log(vfloat3 a);
    vfloat4 log(vfloat4 a);
    double log(double a);
    vdouble2 log(vdouble2 a);
    vdouble3 log(vdouble3 a);
    vdouble4 log(vdouble4 a);
    // color log(color const &a);  (not supported yet)
    float log2(float a);
    vfloat2 log2(vfloat2 a);
    vfloat3 log2(vfloat3 a);
    vfloat4 log2(vfloat4 a);
    double log2(double a);
    vdouble2 log2(vdouble2 a);
    vdouble3 log2(vdouble3 a);
    vdouble4 log2(vdouble4 a);
    // color log2(color const &a);  (not supported yet)
    float log10(float a);
    vfloat2 log10(vfloat2 a);
    vfloat3 log10(vfloat3 a);
    vfloat4 log10(vfloat4 a);
    double log10(double a);
    vdouble2 log10(vdouble2 a);
    vdouble3 log10(vdouble3 a);
    vdouble4 log10(vdouble4 a);
    // color log10(color const &a);  (not supported yet)
    float luminance(vfloat3 a);
    // float luminance(color const &a);  (not supported yet)
    int max(int a, int b);
    vint2 max(vint2 a, vint2 b);
    vint3 max(vint3 a, vint3 b);
    vint4 max(vint4 a, vint4 b);
    float max(float a, float b);
    vfloat2 max(vfloat2 a, vfloat2 b);
    vfloat3 max(vfloat3 a, vfloat3 b);
    vfloat4 max(vfloat4 a, vfloat4 b);
    double max(double a, double b);
    vdouble2 max(vdouble2 a, vdouble2 b);
    vdouble3 max(vdouble3 a, vdouble3 b);
    vdouble4 max(vdouble4 a, vdouble4 b);
    // color max(color const &a, color const &b);  (not supported yet)
    // color max(float a, color const &b);  (not supported yet)
    // color max(color const &a, float b);  (not supported yet)
    float max_value(float a);
    float max_value(vfloat2 a);
    float max_value(vfloat3 a);
    float max_value(vfloat4 a);
    double max_value(double a);
    double max_value(vdouble2 a);
    double max_value(vdouble3 a);
    double max_value(vdouble4 a);
    // float max_value(color const &a);  (not supported yet)
    int min(int a, int b);
    vint2 min(vint2 a, vint2 b);
    vint3 min(vint3 a, vint3 b);
    vint4 min(vint4 a, vint4 b);
    float min(float a, float b);
    vfloat2 min(vfloat2 a, vfloat2 b);
    vfloat3 min(vfloat3 a, vfloat3 b);
    vfloat4 min(vfloat4 a, vfloat4 b);
    double min(double a, double b);
    vdouble2 min(vdouble2 a, vdouble2 b);
    vdouble3 min(vdouble3 a, vdouble3 b);
    vdouble4 min(vdouble4 a, vdouble4 b);
    // color min(color const &a, color const &b);  (not supported yet)
    // color min(float a, color const &b);  (not supported yet)
    // color min(color const &a, float b);  (not supported yet)
    float min_value(float a);
    float min_value(vfloat2 a);
    float min_value(vfloat3 a);
    float min_value(vfloat4 a);
    double min_value(double a);
    double min_value(vdouble2 a);
    double min_value(vdouble3 a);
    double min_value(vdouble4 a);
    // float min_value(color const &a);  (not supported yet)
    // float[2] modf(float a);  (not supported yet)
    // float2[2] modf(vfloat2 a);  (not supported yet)
    // float3[2] modf(vfloat3 a);  (not supported yet)
    // float4[2] modf(vfloat4 a);  (not supported yet)
    // double[2] modf(double a);  (not supported yet)
    // double2[2] modf(vdouble2 a);  (not supported yet)
    // double3[2] modf(vdouble3 a);  (not supported yet)
    // double4[2] modf(vdouble4 a);  (not supported yet)
    float normalize(float a);
    vfloat2 normalize(vfloat2 a);
    vfloat3 normalize(vfloat3 a);
    vfloat4 normalize(vfloat4 a);
    double normalize(double a);
    vdouble2 normalize(vdouble2 a);
    vdouble3 normalize(vdouble3 a);
    vdouble4 normalize(vdouble4 a);
    int pow(int a, int b);
    vint2 pow(vint2 a, vint2 b);
    vint3 pow(vint3 a, vint3 b);
    vint4 pow(vint4 a, vint4 b);
    float pow(float a, float b);
    vfloat2 pow(vfloat2 a, vfloat2 b);
    vfloat3 pow(vfloat3 a, vfloat3 b);
    vfloat4 pow(vfloat4 a, vfloat4 b);
    double pow(double a, double b);
    vdouble2 pow(vdouble2 a, vdouble2 b);
    vdouble3 pow(vdouble3 a, vdouble3 b);
    vdouble4 pow(vdouble4 a, vdouble4 b);
    vint2 pow(vint2 a, int b);
    vint3 pow(vint3 a, int b);
    vint4 pow(vint4 a, int b);
    vfloat2 pow(vfloat2 a, float b);
    vfloat3 pow(vfloat3 a, float b);
    vfloat4 pow(vfloat4 a, float b);
    vdouble2 pow(vdouble2 a, double b);
    vdouble3 pow(vdouble3 a, double b);
    vdouble4 pow(vdouble4 a, double b);
    // color pow(color const &a, color const &b);  (not supported yet)
    // color pow(color const &a, float b);  (not supported yet)
    float radians(float a);
    vfloat2 radians(vfloat2 a);
    vfloat3 radians(vfloat3 a);
    vfloat4 radians(vfloat4 a);
    double radians(double a);
    vdouble2 radians(vdouble2 a);
    vdouble3 radians(vdouble3 a);
    vdouble4 radians(vdouble4 a);
    float round(float a);
    vfloat2 round(vfloat2 a);
    vfloat3 round(vfloat3 a);
    vfloat4 round(vfloat4 a);
    double round(double a);
    vdouble2 round(vdouble2 a);
    vdouble3 round(vdouble3 a);
    vdouble4 round(vdouble4 a);
    float rsqrt(float a);
    vfloat2 rsqrt(vfloat2 a);
    vfloat3 rsqrt(vfloat3 a);
    vfloat4 rsqrt(vfloat4 a);
    double rsqrt(double a);
    vdouble2 rsqrt(vdouble2 a);
    vdouble3 rsqrt(vdouble3 a);
    vdouble4 rsqrt(vdouble4 a);
    // color rsqrt(color const &a);  (not supported yet)
    float saturate(float a);
    vfloat2 saturate(vfloat2 a);
    vfloat3 saturate(vfloat3 a);
    vfloat4 saturate(vfloat4 a);
    double saturate(double a);
    vdouble2 saturate(vdouble2 a);
    vdouble3 saturate(vdouble3 a);
    vdouble4 saturate(vdouble4 a);
    // color saturate(color const &a);  (not supported yet)
    int sign(int a);
    vint2 sign(vint2 a);
    vint3 sign(vint3 a);
    vint4 sign(vint4 a);
    float sign(float a);
    vfloat2 sign(vfloat2 a);
    vfloat3 sign(vfloat3 a);
    vfloat4 sign(vfloat4 a);
    double sign(double a);
    vdouble2 sign(vdouble2 a);
    vdouble3 sign(vdouble3 a);
    vdouble4 sign(vdouble4 a);
    float sin(float a);
    vfloat2 sin(vfloat2 a);
    vfloat3 sin(vfloat3 a);
    vfloat4 sin(vfloat4 a);
    double sin(double a);
    vdouble2 sin(vdouble2 a);
    vdouble3 sin(vdouble3 a);
    vdouble4 sin(vdouble4 a);
    // float[2] sincos(float a);  (not supported yet)
    // float2[2] sincos(vfloat2 a);  (not supported yet)
    // float3[2] sincos(vfloat3 a);  (not supported yet)
    // float4[2] sincos(vfloat4 a);  (not supported yet)
    // double[2] sincos(double a);  (not supported yet)
    // double2[2] sincos(vdouble2 a);  (not supported yet)
    // double3[2] sincos(vdouble3 a);  (not supported yet)
    // double4[2] sincos(vdouble4 a);  (not supported yet)
    float smoothstep(float a, float b, float l);
    vfloat2 smoothstep(vfloat2 a, vfloat2 b, vfloat2 l);
    vfloat3 smoothstep(vfloat3 a, vfloat3 b, vfloat3 l);
    vfloat4 smoothstep(vfloat4 a, vfloat4 b, vfloat4 l);
    double smoothstep(double a, double b, double l);
    vdouble2 smoothstep(vdouble2 a, vdouble2 b, vdouble2 l);
    vdouble3 smoothstep(vdouble3 a, vdouble3 b, vdouble3 l);
    vdouble4 smoothstep(vdouble4 a, vdouble4 b, vdouble4 l);
    vfloat2 smoothstep(vfloat2 a, vfloat2 b, float l);
    vfloat3 smoothstep(vfloat3 a, vfloat3 b, float l);
    vfloat4 smoothstep(vfloat4 a, vfloat4 b, float l);
    vdouble2 smoothstep(vdouble2 a, vdouble2 b, double l);
    vdouble3 smoothstep(vdouble3 a, vdouble3 b, double l);
    vdouble4 smoothstep(vdouble4 a, vdouble4 b, double l);
    float sqrt(float a);
    vfloat2 sqrt(vfloat2 a);
    vfloat3 sqrt(vfloat3 a);
    vfloat4 sqrt(vfloat4 a);
    double sqrt(double a);
    vdouble2 sqrt(vdouble2 a);
    vdouble3 sqrt(vdouble3 a);
    vdouble4 sqrt(vdouble4 a);
    // color sqrt(color const &a);  (not supported yet)
    float step(float a, float b);
    vfloat2 step(vfloat2 a, vfloat2 b);
    vfloat3 step(vfloat3 a, vfloat3 b);
    vfloat4 step(vfloat4 a, vfloat4 b);
    double step(double a, double b);
    vdouble2 step(vdouble2 a, vdouble2 b);
    vdouble3 step(vdouble3 a, vdouble3 b);
    vdouble4 step(vdouble4 a, vdouble4 b);
    float tan(float a);
    vfloat2 tan(vfloat2 a);
    vfloat3 tan(vfloat3 a);
    vfloat4 tan(vfloat4 a);
    double tan(double a);
    vdouble2 tan(vdouble2 a);
    vdouble3 tan(vdouble3 a);
    vdouble4 tan(vdouble4 a);
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
    // color blackbody(float temperature);  (not supported yet)
    // color emission_color(float[<N>] wavelengths, float[N] amplitudes);  (not supported yet)
    // color emission_color(color const &value);  (not supported yet)
}

namespace debug
{
    bool breakpoint();
    bool assert(bool condition, string reason, string funcname = "", string filename = "",
        int line = 0);
    bool print(bool v);
    bool print(vbool2 v);
    bool print(vbool3 v);
    bool print(vbool4 v);
    bool print(int v);
    bool print(vint2 v);
    bool print(vint3 v);
    bool print(vint4 v);
    bool print(float v);
    bool print(vfloat2 v);
    bool print(vfloat3 v);
    bool print(vfloat4 v);
    bool print(double v);
    bool print(vdouble2 v);
    bool print(vdouble3 v);
    bool print(vdouble4 v);
    // bool print(color const &v);  (not supported yet)
    bool print(string v);
}

#endif  // MDL_USER_MODULES_MDL_RUNTIME_H
