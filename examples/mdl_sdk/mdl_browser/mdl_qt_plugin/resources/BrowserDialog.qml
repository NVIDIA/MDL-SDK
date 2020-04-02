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

import QtQuick 2.10
import QtQuick.Controls 2.3
import QtQuick.Controls.Material 2.3
import QtQuick.Dialogs 1.3
import MdlQtPlugin 1.0 as Mdl

Dialog {
    id: id_control
    property string result: ""                  // contains the selected material, function, ...
                                                // or an empty string

    property bool keep_previous_select: false   // keeps the selected object selected 
                                                // when opening the dialog the next time

    width: 1280                                 // default window width
    height: 720                                 // default window height
    visible: false                              // initially hidden, call .open() to show it
    title: "MDL Material Browser Dialog"        // window title that can be overridden 

    QtObject {
        id: id_private
        property bool accepted;
    }

    contentItem: Mdl.BrowserMain {
        id: id_main

        onAccepted: {
            id_private.accepted = true;
            id_control.result = id_main.result;
            id_control.accept();
        }

        onRejected: {
            id_private.accepted = false;
            id_control.result = "";
            id_control.reject();
        }
    }

    onVisibilityChanged: {

        // reset to defaults when the window is shown
        if(visible) {   
            id_private.accepted = false;
            id_control.result = "";

            if(!keep_previous_select)
                id_main.reset();
        }
        // handle closing of the dialog as reject
        else {
            if(!id_private.accepted)
                id_main.reject();
        }
    }
}
