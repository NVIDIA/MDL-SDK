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


#include "selection_sorter.h"
#include "../../cache/imdl_cache.h"

Selection_sorter_base::Selection_sorter_base(const std::string& name, bool ascending)
    : m_name(name)
    , m_ascending(ascending)
{
}

float Selection_sorter_ranking::compare(const VM_sel_element* left, 
                                        const VM_sel_element* right) const
{
    return left->get_search_ranking() - right->get_search_ranking();
}

float Selection_sorter_name::compare(const VM_sel_element* left, 
                                     const VM_sel_element* right) const
{
    QString left_name = left->get_display_name();
    if (left_name.isEmpty()) left_name = left->get_name();

    QString right_name = right->get_display_name();
    if (right_name.isEmpty()) right_name = right->get_name();

    return static_cast<float>(QString::compare(left_name, right_name, Qt::CaseInsensitive));
}

float Selection_sorter_date::compare(const VM_sel_element* left, const VM_sel_element* right) const
{
    const auto parent_module_left = left->get_parent_module_cache_element();
    const auto parent_module_right = right->get_parent_module_cache_element();

    return static_cast<float>(static_cast<int64_t>(parent_module_left->get_timestamp())
           - static_cast<int64_t>(parent_module_right->get_timestamp()));
}
