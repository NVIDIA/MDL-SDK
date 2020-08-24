/******************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/dxr/mdl_d3d12/window.h

#ifndef MDL_D3D12_IWINDOW_H
#define MDL_D3D12_IWINDOW_H

#include "common.h"

namespace mi { namespace examples { namespace gui
{
    class Root;
}}}

namespace mi { namespace examples { namespace mdl_d3d12
{
    class Texture;

    // ------------------------------------------------------------------------

    class IWindow
    {
    public:

        // --------------------------------------------------------------------

        enum class Mode
        {
            Windowed,
            Fullsceen,
            // Borderless_windowed /*not implemented*/
        };

        // --------------------------------------------------------------------

        virtual ~IWindow() = default;

        /// called internally in order to run the main loop
        virtual int show(int nCmdShow) = 0;

        /// closes the window after finishing the current frame
        virtual void close() = 0;

        /// returns true if the window is the foreground window
        virtual bool has_focus() const = 0;

        virtual Texture* get_back_buffer() const = 0;
        virtual D3D12_CPU_DESCRIPTOR_HANDLE get_back_buffer_rtv() const = 0;
        virtual bool present_back_buffer() = 0;


        virtual size_t get_width() const = 0;
        virtual size_t get_height() const = 0;
        virtual bool resize(size_t width, size_t height, double dpi) = 0;

        virtual void set_vsync(bool on) = 0;
        virtual bool get_vsync() const = 0;

        virtual void set_window_mode(Mode mode) = 0;
        virtual Mode get_window_mode() const = 0;

        // get the windows main UI instance
        virtual mi::examples::gui::Root* get_gui() = 0;
    };

}}} // mi::examples::mdl_d3d12
#endif
