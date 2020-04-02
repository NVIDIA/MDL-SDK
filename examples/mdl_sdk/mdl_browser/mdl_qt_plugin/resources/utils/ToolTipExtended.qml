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
import QtQuick.Controls.impl 2.3
import QtQuick.Templates 2.3 as T
import QtQuick.Controls.Material 2.3

T.ToolTip {
    id: id_control
    property int maximumWidth: 300

    x: parent ? (parent.width - id_background.implicitWidth) / 2 : 0
    y: -id_background.implicitHeight - 20
    implicitWidth: id_background.implicitWidth
    implicitHeight: id_background.implicitHeight

    function show() {
        if(id_background.implicitWidth > maximumWidth)
            id_background.implicitWidth = maximumWidth;

        visible = true;
    }

    function hide() {
        if(id_toolTipMouse && id_toolTipMouse.containsMouse) return;
        visible = false;
    }

    margins: 12
    padding: 8
    leftPadding: padding + 8
    rightPadding: padding + 8

    closePolicy: T.Popup.CloseOnEscape | T.Popup.CloseOnPressOutsideParent | T.Popup.CloseOnReleaseOutsideParent

    enter: Transition {
        NumberAnimation { property: "opacity"; from: 0.0; to: 1.0; easing.type: Easing.OutQuad; duration: 500 }
    }

    exit: Transition {
        NumberAnimation { property: "opacity"; from: 1.0; to: 0.0; easing.type: Easing.InQuad; duration: 500 }
    }

    contentItem: Text {
        id: id_text
        text: id_control.text
        font: id_control.font
        wrapMode: Text.WrapAtWordBoundaryOrAnywhere
        color: Material.color(Material.Grey, Material.Shade300)
        
        MouseArea {
            id: id_toolTipMouse
            x: id_background.x - leftPadding
            y: id_background.y - topPadding
            width: id_background.implicitWidth
            height: id_background.implicitHeight
            hoverEnabled: true
            onContainsMouseChanged: { if(!containsMouse) { id_control.visible = false; } }

            /*
            Rectangle {
                color: "#55e93535"
                anchors.fill: parent
            }
            */
        }
    }

    background: Rectangle {
        id: id_background
        color: id_control.Material.tooltipColor
        opacity: 0.9
        radius: 2
        implicitWidth: id_text.paintedWidth + leftPadding + rightPadding
        implicitHeight: id_text.implicitHeight + topPadding + bottomPadding
    }
}
