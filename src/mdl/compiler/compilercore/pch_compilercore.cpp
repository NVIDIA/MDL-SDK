/***************************************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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
/**
   \file
   \brief        Precompiled header generation file.
*/

#ifdef WIN_NT

#include <mi/base/interface_implement.h>
#include <mi/base/iallocator.h>
#include <mi/base/handle.h>

#include <string>
#include <vector>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <list>
#include <utility>

#include <base/system/stlext/i_stlext_concepts.h>

// mdl_compiler specific:
#include <mi/mdl/mdl_annotations.h>
#include <mi/mdl/mdl_archiver.h>
#include <mi/mdl/mdl_code_generators.h>
#include <mi/mdl/mdl_declarations.h>
#include <mi/mdl/mdl_definitions.h>
#include <mi/mdl/mdl_expressions.h>
#include <mi/mdl/mdl_file_resolver.h>
#include <mi/mdl/mdl_file_utils.h>
#include <mi/mdl/mdl_generated_code.h>
#include <mi/mdl/mdl_generated_dag.h>
#include <mi/mdl/mdl_generated_executable.h>
#include <mi/mdl/mdl_positions.h>
#include <mi/mdl/mdl_manifest.h>
#include <mi/mdl/mdl_messages.h>
#include <mi/mdl/mdl_modules.h>
#include <mi/mdl/mdl_names.h>
#include <mi/mdl/mdl_options.h>
#include <mi/mdl/mdl_printers.h>
#include <mi/mdl/mdl_statements.h>
#include <mi/mdl/mdl_streams.h>
#include <mi/mdl/mdl_symbols.h>
#include <mi/mdl/mdl_types.h>
#include <mi/mdl/mdl_values.h>

#endif // WIN_NT

// Include the general precompiled header file AS LAST FILE:
#include "pch.h"
