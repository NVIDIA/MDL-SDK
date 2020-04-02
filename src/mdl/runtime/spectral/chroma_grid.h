/******************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
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
 *****************************************************************************/
/// \file
/// \brief  Chromaticity grid based color to spectrum conversion
///
/// Mostly along the lines of "Meng, Simon, Hanika, Dachsbacher - 
/// Physically meaningful rendering using tristimulus colours":
/// - the conversion utilizes a grid of precomputed spectra on the CIE (x,y) chromaticity plane
/// - mostly this is rotated regular grid, but a the borders of the visible gamut the cells may
///   be a quadrilateral or a triangle instead of a rectangle
/// - the spectra are scaled to unity brightness (X + Y + Z = 1) such that interpolation can be
///   performed on chromaticity
///
/// important changes / extensions (for the data we use)
/// - we respect a whitepoint (illuminant):
///   + we have different grids for E, D60, and D65
///   + the whitepoint coincides with a grid point
/// - the spectra are optimized for smoothness _relative_ to the whitepoint, i.e. such that for 
///   e.g. sRGB the color (1,1,1) can yield D65 for absolute conversion and equispectrum for
///   relative conversion

#ifndef MDL_RUNTIME_SPECTRAL_CHROMA_GRID_H_
#define MDL_RUNTIME_SPECTRAL_CHROMA_GRID_H_


namespace mi {
namespace mdl {
namespace spectral {

enum {
    CELL_FLAG_RECT = 1 << 31,
    CELL_FLAG_ONE_TRI = 1 << 30,
    CELL_FLAG_TWO_TRI = 1 << 29,
    CELL_FLAG_MASK = ~(CELL_FLAG_RECT | CELL_FLAG_ONE_TRI | CELL_FLAG_TWO_TRI)
};


// 2d triangle inclusion test
static inline bool test_tri(
    float *b0, float *b1, float *b2,
    const float x, const float y,
    const float p0[2], const float p1[2], const float p2[2])
{
    const float e1x = p1[0] - p0[0];
    const float e1y = p1[1] - p0[1];

    const float e2x = p2[0] - p0[0];
    const float e2y = p2[1] - p0[1];

    const float dx = x - p0[0];
    const float dy = y - p0[1];

    const float inv_det = 1.0f / (e1x * e2y - e2x * e1y);

    *b1 = inv_det * (e2y * dx - e2x * dy);
    *b2 = inv_det * (e1x * dy - e1y * dx);
    *b0 = 1.0f - *b1 - *b2;

    return (*b0 >= 0.0f && *b1 >= 0.0f && *b2 >= 0.0f);

}

// get indices and weights of chromaticity grid points for a given chromaticity,
// returns the number of points
static inline int get_spectra(
    unsigned int idx[4],
    float w[4],
    const float x, const float y,
    const Chroma_grid_info *const info,
    const Chroma_cell *const cells)
{
    const float px = x - info->p0[0];
    const float py = y - info->p0[1];
        
    const float fx = px * info->tf[0] + py * info->tf[1];
    const float fy = px * info->tf[2] + py * info->tf[3];

    int ix = (int)fx;
    int iy = (int)fy;
    const int cell_res = info->res - 1;
    if (ix < 0 || ix >= cell_res ||
        iy < 0 || iy >= cell_res)
        return 0;

    const unsigned int cell_idx = iy * cell_res + ix;
    const Chroma_cell *cell = &cells[cell_idx];

    const unsigned int idx0 = cell->idx[0] & CELL_FLAG_MASK;

    if (cell->idx[0] & CELL_FLAG_RECT)
    {
        // inner cell, four points, rectangle interpolation
        const float wx = fx - (float)ix;
        const float tx = (1.0f - wx);
        const float wy = fy - (float)iy;
        const float ty = (1.0f - wy);

        w[0] = tx * ty;
        w[1] = wx * ty;
        w[2] = wx * wy;
        w[3] = tx * wy;

        idx[0] = idx0;
        idx[1] = cell->idx[1];
        idx[2] = cell->idx[2];
        idx[3] = cell->idx[3];

        return 4;
    }
    else if ((cell->idx[0] & CELL_FLAG_TWO_TRI) || (cell->idx[0] & CELL_FLAG_ONE_TRI))
    {
        // one or two triangles, three points, barycentric interpolation
        if (test_tri(w, w + 1, w + 2, fx, fy, cell->xy, cell->xy + 2, cell->xy + 4))
        {

            idx[0] = idx0;
            idx[1] = cell->idx[1];
            idx[2] = cell->idx[2];
            return 3;
        }
        else if ((cell->idx[0] & CELL_FLAG_TWO_TRI) &&
                test_tri(w, w + 1, w + 2, fx, fy, cell->xy, cell->xy + 4, cell->xy + 6))
        {
            idx[0] = idx0;
            idx[1] = cell->idx[2];
            idx[2] = cell->idx[3];    
            return 3;
        }
    }

    return 0;
}

} // namespace spectral
} // namespace mdl
} // namespace mi

#endif // MDL_RUNTIME_SPECTRAL_CHROMA_GRID_H_
