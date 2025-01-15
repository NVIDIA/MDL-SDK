/******************************************************************************
 * Copyright (c) 2017-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \file framework.h
/// \brief Framework to schedule and execute rule sets on MDL expressions

#pragma once

#include <mi/mdl_sdk.h>

// Helper classes
#include "mdl_assert.h"

// MDL projection classes
#include "options.h"

using mi::neuraylib::INeuray;
using mi::neuraylib::IValue;
using mi::neuraylib::IExpression_direct_call;
using mi::neuraylib::ITransaction;
using mi::neuraylib::IMdl_impexp_api;
using mi::neuraylib::ICompiled_material;

/// Main framework that offers high level methods to convert a compiled MDL material to
/// an MDL expression, applies rules sets, and prints the result in MDL syntax.
/// Multiple frameworks can be used at the same time.
/// TODO: more functions to access baked results etc.
///
class Mdl_projection_framework {
    // prevent the copy constructor from working
    Mdl_projection_framework( const Mdl_projection_framework& fw);
    // prevent the assignment operator from working
    Mdl_projection_framework& operator=( const Mdl_projection_framework&);

public:
    /// Constructs internal MDL expression representation from compiled material.
    Mdl_projection_framework();
};

/// Main function to project an MDL material.
/// Uses an instance compiled material in an ICompiled_material as input,
/// converts it to an MDL expression, applies selected rule sets, optionally
/// bakes function call graphs into textures, and returns the result
/// as a new ICompiled_material. Writes the resulting MDL modules to 'out'
/// if non-null and writes generated textures to disc.
/// It also checks whether the generated modules compile.
///
/// May require adaption to particular use cases.
///
/// \return 0 in case of a failure or if options->test_module is set
///           to false. Otherwise it returns the new ICompiled_material with 
///           a reference count of 1.
///
const ICompiled_material* mdl_distill( INeuray* neuray, // only used in printer.cpp:strip_path()
                                       IMdl_impexp_api* mdl_impexp_api,
                                       ITransaction* transaction,
                                       const ICompiled_material* compiled_material,
                                       const char* material_name,
                                       const char* target,
                                       Options* options,
                                       double add_to_total_time = 0,
                                       std::ostream* out = 0);
