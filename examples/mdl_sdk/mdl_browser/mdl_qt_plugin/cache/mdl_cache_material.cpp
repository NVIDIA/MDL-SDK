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


#include "mdl_cache_material.h"
#include <mi/mdl_sdk.h>

bool Mdl_cache_material::update(mi::neuraylib::INeuray* neuray, 
                                mi::neuraylib::ITransaction* transaction,
                                const mi::base::IInterface* module)
{
    // Access the material definition
    const mi::base::Handle<const mi::neuraylib::IMaterial_definition> material_definition(
        transaction->access<mi::neuraylib::IMaterial_definition>(
            (std::string("mdl") + get_qualified_name()).c_str()));

    if (!material_definition)
    {
        std::cerr << "[Mdl_cache_material] update: Failed to get material defintion: "
                  << get_qualified_name() << "\n";
        return false;
    }

    // get infos from annotations
    const mi::base::Handle<const mi::neuraylib::IAnnotation_block> anno_block(
        material_definition->get_annotations());

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

        const mi::Size ai = annotations.get_annotation_index("::anno::key_words(string[N])");
        if (ai != static_cast<mi::Size>(-1))
        {
            const mi::base::Handle<const mi::neuraylib::IValue> ivalue(
                annotations.get_annotation_param_value(ai, 0));

            const mi::base::Handle<const mi::neuraylib::IValue_array> ivalue_array(
                ivalue->get_interface<const mi::neuraylib::IValue_array>());

            const mi::Size keyword_count = ivalue_array->get_size();
            std::stringstream s;
            for (mi::Size i = 0; i < keyword_count; ++i)
            {
                if (i > 0) s << ", ";

                const mi::base::Handle<const mi::neuraylib::IValue> keyword_value(
                    ivalue_array->get_value<const mi::neuraylib::IValue>(i));

                const mi::base::Handle<const mi::neuraylib::IValue_string> keyword(
                    keyword_value->get_interface<const mi::neuraylib::IValue_string>());

                s << keyword->get_value();
            }

            if (keyword_count > 0)
            {
                std::string keywords = s.str();
                set_cache_data("Keywords", keywords.c_str());
            }
        }
    }

    // would be nice to not use the cache in order to get updates
    // however, loading the material is no option during runtime
    const char* path = material_definition->get_thumbnail();
    if (path)
        set_cache_data("Thumbnail", path);
    
    return true;
}