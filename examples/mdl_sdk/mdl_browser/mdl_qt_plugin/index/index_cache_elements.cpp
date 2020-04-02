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


#include "index_cache_elements.h"
#include "../cache/imdl_cache.h"
#include "../cache/mdl_cache.h"
#include <iostream>


bool build_recursively(Index_cache_elements* index, const IMdl_cache_node* node)
{
    bool success = true;


    for (mi::Size i = 0, n = node->get_child_count(); i<n; ++i)
    {
        // nodes that can have "grand" children  (packages only)
        const IMdl_cache_item* child_item = node->get_child(*node->get_child_key(i));
        const auto child_node = dynamic_cast<const IMdl_cache_node*>(child_item);
        if (child_node)
        {
            build_recursively(index, child_node);
            continue;
        }

        // elements that can be selected
        const auto child_element = dynamic_cast<const IMdl_cache_element*>(child_item);
        if (child_element)
        {
            index->add_document(new Index_document_cache_element(child_element));
            continue;
        }

        std::cerr << "[Index_mdl_cache] build_recursively: missing case.\n";
        success = false;
        break;
    }
    return success;
}

bool Index_cache_elements::build(const Mdl_cache* cache)
{
    const IMdl_cache_package* root = cache->get_cache_root();
    return build_recursively(this, root);
}
