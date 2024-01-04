/***************************************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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

 /// \file
 /// \brief The .axf importer options.

#ifndef EXAMPLE_AXF_TO_MDL_AXF_IMPORTER_OPTIONS_H
#define EXAMPLE_AXF_TO_MDL_AXF_IMPORTER_OPTIONS_H

namespace mi { namespace examples { namespace impaxf {

// Command line options structure.
struct Axf_importer_options
{
    // mdl output filename.
    std::string mdl_output_filename;

    std::string axf_module_prefix;
    std::string axf_color_space;
    std::string axf_color_representation;
    std::vector<std::string> mdl_paths;
    bool nostdpath;
    
    Axf_importer_options()
        : mdl_output_filename("example_axf_to_mdl.mdl")
        , axf_module_prefix("axf")
        , axf_color_space("sRGB,E")
        , axf_color_representation("all")
        , mdl_paths()
        , nostdpath(false)
    {}
};

}}}
#endif