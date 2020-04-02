/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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


#include "mdl_cache_function.h"
#include <mi/mdl_sdk.h>

bool Mdl_cache_function::update(mi::neuraylib::INeuray* neuray, 
                                mi::neuraylib::ITransaction* transaction, 
                                const mi::base::IInterface* module)
{
    // Access the material definition
    const mi::base::Handle<const mi::neuraylib::IFunction_definition> function_definition(
        transaction->access<mi::neuraylib::IFunction_definition>(
            (std::string("mdl") + get_qualified_name()).c_str()));

    if (!function_definition)
    {
        std::cerr << "[Mdl_cache_function] update: Failed to get function definition: "
                  << get_qualified_name() << "\n";
        return false;
    }

    // get infos from annotations
    const mi::base::Handle<const mi::neuraylib::IAnnotation_block> anno_block(
        function_definition->get_annotations());

    // are there annotations?
    if (anno_block)
    {
        const mi::neuraylib::Annotation_wrapper annotations(anno_block.get());
        const char* value = nullptr;

        if (annotations.get_annotation_index("::anno::hidden()") != static_cast<mi::Size>(-1))
            set_is_hidden(true);

        if (0 == annotations.get_annotation_param_value_by_name<const char*>(
            "::anno::author(string)", 0, value))
            set_cache_data("Author", value);

        if (0 == annotations.get_annotation_param_value_by_name<const char*>(
            "::anno::display_name(string)", 0, value))
            set_cache_data("DisplayName", value);

        if (0 == annotations.get_annotation_param_value_by_name<const char*>(
            "::anno::description(string)", 0, value))
            set_cache_data("Description", value);
    }

    return true;
}