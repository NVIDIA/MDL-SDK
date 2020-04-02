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
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0


Item {
    id: id_control
    width: 32
    height: 32

    property string file_path: ""           // resource files, assuming svg
    property string full_file_path: ""      // full path file:///path/file.extension (overrides the file_path)
    property color color: Material.foreground
    property bool hovered: id_icon_mouse.containsMouse
    property bool clickable: true

    signal clicked(var mouse)

    Image {
        id: id_image
        anchors.fill: parent
        width: parent.width
        height: parent.height

        // general
        fillMode: Image.PreserveAspectFit
    
        // vector graphics
        sourceSize.width: width
        sourceSize.height: height
        source: (full_file_path != "") 
            ? full_file_path                // full_file_path overrides the composed one
            : file_path + ".svg"
        mipmap: true
        asynchronous: true
    }

    ColorOverlay {
        id: id_tint;
        anchors.fill: parent
        source: id_image
        color: id_control.color
    }

    MouseArea {
        id: id_icon_mouse
        enabled: clickable
        anchors.fill: parent
        hoverEnabled: true

        onClicked: id_control.clicked(mouse)
    }
}

