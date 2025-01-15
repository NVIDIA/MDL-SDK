/******************************************************************************
 * Copyright (c) 2012-2025, NVIDIA CORPORATION. All rights reserved.
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
 ******************************************************************************/
#ifndef MDL_COMPILERCORE_BUILDER_H
#define MDL_COMPILERCORE_BUILDER_H

namespace mi {

namespace mdl {
class Module;
class Type_cache;
class Type_factory;
class MDL;

/// Enter all compiler known definitions from compilercore_known_defs.h
/// into the given module.
///
/// \param module                   the module
/// \param tc                       the type cache
/// \param extra_types_are_uniform  true: eXtra modifier means uniform, false eXtra means NONE
void enter_predefined_entities(
    Module     &module,
    Type_cache &tc,
    bool       extra_types_are_uniform);

/// Enter all predefined types into the given type factory.
///
/// \param tf                       the type factory
/// \param extra_types_are_uniform  true: eXtra modifier means uniform, false eXtra means NONE
void enter_predefined_types(
    Type_factory &tf,
    bool         extra_types_are_uniform);

} // mdl
} // mi

#endif // MDL_COMPILERCORE_BUILDER_H
