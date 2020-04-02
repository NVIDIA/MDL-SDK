/***************************************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "libmdlrt.h"

#define NULL 0

namespace math
{
    float clamp(float a, float min, float max);

    float exp(float a);

    float floor(float a);

    float max(float a, float b);
}

/// known color spaces
enum Color_space_id {
    CS_XYZ,     // CIE XYZ
    CS_sRGB,    // (linear) sRGB, aka rec709 (HDTV)
    CS_ACES,    // Academy Color Encoding System
    CS_Rec2020  // U-HDTV
};

// CIE standard observer color matching functions (1931)
static const unsigned int SPECTRAL_XYZ_RES  = 81;
static const float SPECTRAL_XYZ_LAMBDA_MIN = 380.000000f;
static const float SPECTRAL_XYZ_LAMBDA_MAX = 780.000000f;
static const float SPECTRAL_XYZ_LAMBDA_STEP = 5.0f;

/* CIE standard observer color matching functions (1931)*/
static const float SPECTRAL_XYZ1931_X[] = {
    0.0013680001f, 0.0022360000f, 0.0042429999f, 0.0076500000f, 0.0143100005f, 0.0231899992f,
    0.0435100012f, 0.0776299983f, 0.1343799978f, 0.2147700042f, 0.2838999927f, 0.3285000026f,
    0.3482800126f, 0.3480600119f, 0.3361999989f, 0.3186999857f, 0.2908000052f, 0.2511000037f,
    0.1953600049f, 0.1421000063f, 0.0956400037f, 0.0579500012f, 0.0320100002f, 0.0147000002f,
    0.0049000001f, 0.0024000001f, 0.0093000000f, 0.0291000009f, 0.0632700026f, 0.1096000001f,
    0.1655000001f, 0.2257499993f, 0.2903999984f, 0.3596999943f, 0.4334500134f, 0.5120499730f,
    0.5945000052f, 0.6783999801f, 0.7620999813f, 0.8424999714f, 0.9162999988f, 0.9786000252f,
    1.0262999535f, 1.0566999912f, 1.0621999502f, 1.0456000566f, 1.0025999546f, 0.9383999705f,
    0.8544499874f, 0.7513999939f, 0.6424000263f, 0.5418999791f, 0.4478999972f, 0.3607999980f,
    0.2834999859f, 0.2187000066f, 0.1649000049f, 0.1212000027f, 0.0873999968f, 0.0636000037f,
    0.0467699990f, 0.0329000019f, 0.0227000006f, 0.0158399995f, 0.0113589996f, 0.0081110001f,
    0.0057899999f, 0.0041089999f, 0.0028990000f, 0.0020490000f, 0.0014400000f, 0.0010000000f,
    0.0006900000f, 0.0004760000f, 0.0003320000f, 0.0002350000f, 0.0001660000f, 0.0001170000f,
    0.0000830000f, 0.0000590000f, 0.0000420000f };

static const float SPECTRAL_XYZ1931_Y[] = {
    0.0000390000f, 0.0000640000f, 0.0001200000f, 0.0002170000f, 0.0003960000f, 0.0006400000f,
    0.0012100000f, 0.0021800001f, 0.0040000002f, 0.0073000002f, 0.0115999999f, 0.0168399997f,
    0.0230000000f, 0.0297999997f, 0.0379999988f, 0.0480000004f, 0.0599999987f, 0.0738999993f,
    0.0909800008f, 0.1125999987f, 0.1390199959f, 0.1693000048f, 0.2080200016f, 0.2585999966f,
    0.3230000138f, 0.4072999954f, 0.5030000210f, 0.6082000136f, 0.7099999785f, 0.7932000160f,
    0.8619999886f, 0.9148499966f, 0.9539999962f, 0.9803000093f, 0.9949499965f, 1.0000000000f,
    0.9950000048f, 0.9786000252f, 0.9520000219f, 0.9154000282f, 0.8700000048f, 0.8162999749f,
    0.7570000291f, 0.6948999763f, 0.6309999824f, 0.5667999983f, 0.5030000210f, 0.4411999881f,
    0.3810000122f, 0.3210000098f, 0.2649999857f, 0.2169999927f, 0.1749999970f, 0.1381999999f,
    0.1070000008f, 0.0816000029f, 0.0610000007f, 0.0445800014f, 0.0320000015f, 0.0231999997f,
    0.0170000009f, 0.0119200004f, 0.0082099997f, 0.0057230000f, 0.0041020000f, 0.0029290000f,
    0.0020910001f, 0.0014840000f, 0.0010470001f, 0.0007400000f, 0.0005200000f, 0.0003610000f,
    0.0002490000f, 0.0001720000f, 0.0001200000f, 0.0000850000f, 0.0000600000f, 0.0000420000f,
    0.0000300000f, 0.0000210000f, 0.0000150000f };

static const float SPECTRAL_XYZ1931_Z[] = {
    0.0064500002f, 0.0105499998f, 0.0200500004f, 0.0362100005f, 0.0678500012f, 0.1102000028f,
    0.2073999941f, 0.3713000119f, 0.6456000209f, 1.0390499830f, 1.3855999708f, 1.6229599714f,
    1.7470599413f, 1.7826000452f, 1.7721099854f, 1.7440999746f, 1.6691999435f, 1.5281000137f,
    1.2876399755f, 1.0419000387f, 0.8129500151f, 0.6161999702f, 0.4651800096f, 0.3533000052f,
    0.2720000148f, 0.2123000026f, 0.1581999958f, 0.1116999984f, 0.0782499984f, 0.0572500005f,
    0.0421600007f, 0.0298400000f, 0.0203000009f, 0.0133999996f, 0.0087500000f, 0.0057500000f,
    0.0038999999f, 0.0027500000f, 0.0020999999f, 0.0018000000f, 0.0016500000f, 0.0014000000f,
    0.0011000000f, 0.0010000000f, 0.0008000000f, 0.0006000000f, 0.0003400000f, 0.0002400000f,
    0.0001900000f, 0.0001000000f, 0.0000500000f, 0.0000300000f, 0.0000200000f, 0.0000100000f,
    0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
    0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
    0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
    0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
    0.0000000000f, 0.0000000000f, 0.0000000000f,
    1.f /* dummy non-zero value to stop Clang from emitting a struct of two arrays */ };

static float const tf_xyz_to_srgb[] = {
    3.240600f, -1.537200f, -0.498600f,
    -0.968900f,  1.875800f,  0.041500f,
    0.055700f, -0.204000f,  1.057000f
};

static float const tf_xyz_to_rec2020[] = {
    1.7166511879712680f, -0.3556707837763924f, -0.2533662813736599f,
    -0.6666843518324890f,  1.6164812366349390f,  0.0157685458139111f,
    0.0176398574453108f, -0.0427706132578085f,  0.9421031212354738f
};

static float const tf_xyz_to_aces[] = {
    1.049811017497974f,  0.000000000000000f, -0.000097484540579f,
    -0.495903023077320f,  1.373313045815706f,  0.098240036057310f,
    0.000000000000000f,  0.000000000000000f,  0.991252018200499f
};

static const float D60[SPECTRAL_XYZ_RES] = {
    41.1939622845311f, 43.7996495741081f, 46.4053443694044f, 59.2495668772834f,
    72.0937893734314f, 76.1604005037247f, 80.2270095879923f, 81.4725118293678f,
    82.7180196233373f, 80.1213189543133f, 77.5246259457215f, 86.5451852785837f,
    95.5657542869197f,101.7081387279023f,107.8505211229663f,108.6595358061121f,
    109.4685429835387f,108.5770405324361f,107.6855389805656f,108.6509486391944f,
    109.6163593208360f,106.6350700543170f,103.6537816869229f,104.4217965388071f,
    105.1898027381265f,104.7924104156134f,104.3950252110957f,103.4526535006373f,
    102.5102812786188f,104.2788070205345f,106.0473325067221f,104.6771051706626f,
    103.3068780903832f,103.4217635190818f,103.5366566411065f,101.7683245058292f,
    100.0000000000000f, 98.3771714612798f, 96.7543429225596f, 96.7360183925496f,
    96.7176862891862f, 93.3024888291968f, 89.8872913692075f, 90.9186373938613f,
    91.9499759127423f, 91.9918659750474f, 92.0337561611333f, 91.3034898296861f,
    90.5732313752485f, 88.5107796925075f, 86.4483358868298f, 86.9585785011902f,
    87.4688137335067f, 85.6594825178889f, 83.8501507908184f, 84.2114221156823f,
    84.5726924341244f, 85.9476350700216f, 87.3225772110033f, 85.3115072039523f,
    83.3004448264548f, 78.6644159193148f, 74.0283946417285f, 75.2396639291860f,
    76.4509408627346f, 77.6791169718379f, 78.9073001988292f, 72.1297552280172f,
    65.3522140720051f, 69.6646851424460f, 73.9771526456446f, 76.6842559206226f,
    79.3913599546182f, 73.2923675079999f, 67.1933750613297f, 58.1890199083549f,
    49.1846609406801f, 59.9755410371265f, 70.7664175661787f, 68.9075859336080f,
    67.0487459125749f };

static const float D65[SPECTRAL_XYZ_RES] = {
    49.9754981995000f, 52.3117980957000f, 54.6482009888000f, 68.7014999390000f,
    82.7548980713000f, 87.1203994751000f, 91.4860000610000f, 92.4589004517000f,
    93.4318008423000f, 90.0569992065000f, 86.6822967529000f, 95.7735977173000f,
    104.8649978638000f,110.9359970093000f,117.0080032349000f,117.4100036621000f,
    117.8119964600000f,116.3359985352000f,114.8610000610000f,115.3919982910000f,
    115.9229965210000f,112.3669967651000f,108.8109970093000f,109.0820007324000f,
    109.3539962769000f,108.5780029297000f,107.8020019531000f,106.2959976196000f,
    104.7900009155000f,106.2389984131000f,107.6890029907000f,106.0469970703000f,
    104.4049987793000f,104.2249984741000f,104.0459976196000f,102.0230026245000f,
    100.0000000000000f, 98.1670989990000f, 96.3341979980000f, 96.0610961914000f,
    95.7880020142000f, 92.2368011475000f, 88.6856002808000f, 89.3459014893000f,
    90.0062026978000f, 89.8025970459000f, 89.5990982056000f, 88.6489028931000f,
    87.6986999512000f, 85.4935989380000f, 83.2885971069000f, 83.4938964844000f,
    83.6992034912000f, 81.8629989624000f, 80.0268020630000f, 80.1206970215000f,
    80.2145996094000f, 81.2462005615000f, 82.2778015137000f, 80.2809982300000f,
    78.2842025757000f, 74.0027008057000f, 69.7212982178000f, 70.6651992798000f,
    71.6091003418000f, 72.9789962769000f, 74.3489990234000f, 67.9765014648000f,
    61.6040000916000f, 65.7447967529000f, 69.8855972290000f, 72.4862976074000f,
    75.0869979858000f, 69.3397979736000f, 63.5927009583000f, 55.0054016113000f,
    46.4182014465000f, 56.6118011475000f, 66.8053970337000f, 65.0941009521000f,
    63.3828010559000f };

//  blackbody emitter, compute intensity at specific wavelength (in nm)
//  and temperature (in Kelvin) using Planck's law 
static float blackbody(const float lambda, const float temperature)
{
    const float c = 2.9979e14f; // speed of light (um / s)
    const float h = 6.626e-22f; // Planck constant (scaled to um^2)
    const float k = 1.38e-11f;  // Boltzmann constant (scaled to um^2)

                                // nm -> um
    const float x = lambda * 1e-3f;

    const float f = 2.0f * h * c * c / (x * x * x * x * x);

    return f / (math::exp(h * c / (x * k * temperature)) - 1.0f);
}

static const float *get_XYZ_to_cs(const Color_space_id cs)
{
    switch (cs)
    {
    case CS_XYZ:
        return NULL;
    case CS_sRGB:
        return tf_xyz_to_srgb;
    case CS_ACES:
        return tf_xyz_to_aces;
    case CS_Rec2020:
        return tf_xyz_to_rec2020;
    }
    return NULL;
}

static void convert_XYZ_to_cs(float target[3], const float source[3], const Color_space_id cs)
{
    const float *const m = get_XYZ_to_cs(cs);
    if (!cs)
    {
        target[0] = source[0];
        target[1] = source[1];
        target[2] = source[2];
        return;
    }

    target[0] = source[0] * m[0];
    target[1] = source[0] * m[3];
    target[2] = source[0] * m[6];
    for (unsigned int i = 1; i < 3; ++i) {
        target[0] += source[i] * m[i];
        target[1] += source[i] * m[3 + i];
        target[2] += source[i] * m[6 + i];
    }
}

// get value from a spectrum using linear interpolation
static float get_value_lerp(
    const float * const table_values,
    const unsigned int num_values,
    const float lambda_min,
    const float lambda_max,
    const float lambda)
{
    const float f = (lambda - lambda_min) / (lambda_max - lambda_min) * (float)(num_values - 1);
    unsigned int b0 = (unsigned int)(math::max(math::floor(f), 0.0f));
    if (b0 >= num_values)
        b0 = num_values - 1;
    const unsigned int b1 = (b0 == num_values - 1) ? b0 : b0 + 1;

    const float f1 = f - (float)b0;
    return table_values[b0] * (1.0f - f1) + table_values[b1] * f1;
}

// helper for function below
static float get_value_incremental(
    float const wavelength[],
    float const amplitudes[],
    unsigned   num_values,
    unsigned   &search_pos,
    float      lambda)
{
    unsigned int p1 = search_pos;
    while (p1 < num_values && lambda > wavelength[p1]) {
        ++p1;
    }
    const unsigned int p0 = p1 > 0 ? p1 - 1 : p1;
    search_pos = p0;
    if (p1 >= num_values)
        p1 = num_values - 1;

    if (p0 == p1)
        return amplitudes[p0];

    const float w0 = (lambda - wavelength[p0]) / (wavelength[p1] - wavelength[p0]);
    return
        amplitudes[p0] * w0 + amplitudes[p1] * (1.0f - w0);
}

static void spectrum_to_XYZ(
    float       XYZ[3],
    float const wavelenghts[],
    float const amplitudes[],
    unsigned    num_values)
{
    XYZ[0] = 0.0f;
    XYZ[1] = 0.0f;
    XYZ[2] = 0.0f;

    unsigned search_pos = 0;
    for (unsigned i = 0; i < SPECTRAL_XYZ_RES; ++i) {
        const float lambda = SPECTRAL_XYZ_LAMBDA_MIN + (float)i * SPECTRAL_XYZ_LAMBDA_STEP;

        const float val = get_value_incremental(
            wavelenghts, amplitudes, num_values, search_pos, lambda);
        XYZ[0] += SPECTRAL_XYZ1931_X[i] * val;
        XYZ[1] += SPECTRAL_XYZ1931_Y[i] * val;
        XYZ[2] += SPECTRAL_XYZ1931_Z[i] * val;
    }

    const float scale = (float)(683.002) * SPECTRAL_XYZ_LAMBDA_STEP;
    XYZ[0] *= scale;
    XYZ[1] *= scale;
    XYZ[2] *= scale;
}

static void spectrum_to_cs_refl(
    float       refl[3],
    float const wavelenghts[],
    float const amplitudes[],
    unsigned    num_lambda,
    const Color_space_id cs)
{
    // use white point of color space as illuminant spectrum
    //!! TODO: make illuminant configurable?
    const float *illuminant = NULL;
    switch (cs) {
    case CS_XYZ:
        illuminant = NULL; // E
        break;
    case CS_sRGB:
    case CS_Rec2020:
        illuminant = D65;
        break;
    case CS_ACES:
        illuminant = D60;
        break;
    }

    // compute true spectral multiplication result converted to XYZ
    float XYZ_spectral[3] = { 0.0f, 0.0f, 0.0f };
    // and compute illuminant in converted to XYZ
    float XYZ_illum[3] = { 0.0f, 0.0f, 0.0f };

    unsigned search_pos = 0;
    for (unsigned int i = 0; i < SPECTRAL_XYZ_RES; ++i) {
        const float lambda = SPECTRAL_XYZ_LAMBDA_MIN + (float)i * SPECTRAL_XYZ_LAMBDA_STEP;


        const float illum = illuminant ? illuminant[i] : 1.0f;

        const float val = get_value_incremental(
            wavelenghts, amplitudes, num_lambda, search_pos, lambda) * illum;

        XYZ_spectral[0] += SPECTRAL_XYZ1931_X[i] * val;
        XYZ_spectral[1] += SPECTRAL_XYZ1931_Y[i] * val;
        XYZ_spectral[2] += SPECTRAL_XYZ1931_Z[i] * val;

        XYZ_illum[0] += SPECTRAL_XYZ1931_X[i] * illum;
        XYZ_illum[1] += SPECTRAL_XYZ1931_Y[i] * illum;
        XYZ_illum[2] += SPECTRAL_XYZ1931_Z[i] * illum;
    }
    // note: can ignore actual scaling for both integrals (since it's identical)

    // convert both to the color space
    float cs_spectral[3];
    convert_XYZ_to_cs(cs_spectral, XYZ_spectral, cs);
    float cs_illum[3];
    convert_XYZ_to_cs(cs_illum, XYZ_illum, cs);

    // compute corresponding color space reflectivity (note: can be < 0.0 and > 1.0)
    refl[0] = cs_spectral[0] / cs_illum[0];
    refl[1] = cs_spectral[1] / cs_illum[1];
    refl[2] = cs_spectral[2] / cs_illum[2];
}

extern "C" void mdl_emission_color(
    float       target[3],
    float const wavelenghts[],
    float const amplitudes[],
    unsigned    num_values)
{
    float XYZ[3];

    if (num_values == 0) {
        target[0] = target[1] = target[2] = 0.0f;
        return;
    }

    spectrum_to_XYZ(
        XYZ,
        wavelenghts,
        amplitudes,
        num_values);
    convert_XYZ_to_cs(target, XYZ, CS_sRGB);

    math::max(target[0], 0.0f);
    math::max(target[1], 0.0f);
    math::max(target[2], 0.0f);
}

extern "C" void mdl_reflection_color(
    float       target[3],
    float const wavelenghts[],
    float const amplitudes[],
    unsigned    num_values)
{
    float XYZ[3];

    if (num_values == 0) {
        target[0] = target[1] = target[2] = 0.0f;
        return;
    }

    spectrum_to_cs_refl(
        target,
        wavelenghts,
        amplitudes,
        num_values,
        CS_sRGB);

    target[0] = math::max(target[0], 0.0f);
    target[1] = math::max(target[1], 0.0f);
    target[2] = math::max(target[2], 0.0f);
}

extern "C" void mdl_blackbody(float sRGB[3], float kelvin)
{
    const float threshold = 500.0f;
    if (kelvin < threshold)
        kelvin = threshold;

    float XYZ[3] = { 0.0f, 0.0f, 0.0f };

    // code currently operates on full resolution of our tabulated color matching functions
    for (unsigned int i = 0; i < SPECTRAL_XYZ_RES; ++i)
    {
        const float lambda = SPECTRAL_XYZ_LAMBDA_MIN + (float)i * SPECTRAL_XYZ_LAMBDA_STEP;
        const float f = blackbody(lambda, kelvin);

        XYZ[0] += SPECTRAL_XYZ1931_X[i] * f;
        XYZ[1] += SPECTRAL_XYZ1931_Y[i] * f;
        XYZ[2] += SPECTRAL_XYZ1931_Z[i] * f;
    }

    float scale = 1.0f / XYZ[1];
    XYZ[0] *= scale;
    XYZ[2] *= scale;
    XYZ[1] = 1.0f;

    convert_XYZ_to_cs(sRGB, XYZ, CS_sRGB);

    sRGB[0] = math::max(sRGB[0], 0.0f);
    sRGB[1] = math::max(sRGB[1], 0.0f);
    sRGB[2] = math::max(sRGB[2], 0.0f);
}
