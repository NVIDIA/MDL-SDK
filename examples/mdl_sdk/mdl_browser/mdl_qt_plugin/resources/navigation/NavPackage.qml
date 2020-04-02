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

import "../utils" as Utils

ItemDelegate {
    id: id_control
    //width: parent.width
    height: 40

    property bool showPresentationCounter: true
    property bool showSepBar: true

    property bool isSelected: true

    padding: 0

    contentItem: Item {
        anchors.fill: parent

        Rectangle {
            id: id_sepBar
            visible: showSepBar
            anchors.top: parent.top
            anchors.topMargin: -1
            anchors.left: parent.left
            anchors.right: parent.right
            height: 1
            color: Material.primary
        }

        Utils.Icon {
            id: id_icon
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: parent.left
            anchors.leftMargin: 5
            width: 16
            height: 16

            file_path: packageIsModule
                ? "../graphics/module_icon"
                : "../graphics/package_icon"
            color: isSelected
                ? Material.accent
                : Material.foreground

            onHoveredChanged: {
                if(!moduleIsShadowing) return;
                if(hovered) id_toolTip.show();
                else id_toolTip.hide();
            }

            Utils.ToolTipExtended {
                id: id_toolTip
                delay: 500
                text:  moduleShadowHint
                maximumWidth: 450
                font.pointSize: 10
            }
        }

        // shadow !
        Label {
            id: id_shadowing
            color: "#FF6644"
            visible: moduleIsShadowing
            anchors.bottom: id_icon.bottom
            anchors.bottomMargin: 3
            anchors.left: id_icon.left
            anchors.leftMargin: 9
            text: "!"
            font.pointSize: 14
            font.bold: true
        }

        Label {
            id: id_name
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: id_icon.right
            anchors.leftMargin: 10
            anchors.right: parent.right
            anchors.rightMargin: id_selectionBar.width + 5
            text: packageName
            font.pointSize: 10
            font.bold: true
            elide: Text.ElideRight
        }

        Label {
            visible: id_control.showPresentationCounter
            anchors.top: parent.top
            anchors.topMargin: 3
            anchors.right: parent.right
            anchors.rightMargin: 10
            text: packagePresentationCount
            font.pointSize: 7.5
            horizontalAlignment: Text.AlignRight
        }

        Rectangle {
            id: id_selectionBar
            visible: isSelected
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            anchors.right: parent.right
            color: Material.accent
            width: 3
        }
    }
}
