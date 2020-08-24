/******************************************************************************
* Copyright 2020 NVIDIA Corporation. All rights reserved.
*****************************************************************************/

// a lot of things we need can be found here:
// https://docs.arnoldrenderer.com/display/A5ARP/BSDFs

#ifndef MDL_ARNOLD_H
#define MDL_ARNOLD_H

//-------------------------------------------------------------------------------------------------
// compile time setup
//-------------------------------------------------------------------------------------------------

// enable texture derivatives
#define ENABLE_DERIVATIVES

// number of 8byte cache blocks filled when calling 'init' and used for 'evaluate' and 'sample'
#define NUM_TEXTURE_RESULTS 16

// avoid artifacts when the shading normal differs significantly from the smooth surface normal
// #define APPLY_BUMP_SHADOW_WEIGHT

#endif
