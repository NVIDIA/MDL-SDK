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

 Item{
    id: id_control
    width: id_button.implicitHeight
    height: id_button.implicitWidth

    property string text: "Navigation"
    property bool isCurrent: false

    signal clicked
    //clip: true

    ItemDelegate {
        id: id_button
        anchors.top: parent.top
        implicitWidth: id_label.implicitWidth + 20
        implicitHeight: id_label.implicitHeight + 8
        padding: 0
        z:1

        transform: [
            Rotation { origin.x: 0; origin.y: 0; angle: 90},
            Translate { x: id_button.implicitHeight }
        ]

        Rectangle {
            id: id_background
            anchors.fill: parent
            color: Material.background
        }

        InnerShadow {
            visible: !id_control.isCurrent
            anchors.fill: id_background
            horizontalOffset: 0
            verticalOffset: -3
            radius: 12.0
            samples: 16
            color: "#80000000"
            source: id_background
        }

        Rectangle {
            visible: id_control.isCurrent
            color: Material.accent
            anchors.top: id_button.top
            width: id_button.width
            height: 3
        }

        Label {
            id: id_label
            anchors.bottom: id_button.bottom
            anchors.bottomMargin: 3
            anchors.horizontalCenter: id_button.horizontalCenter
            text: id_control.text.toUpperCase()
            font.pointSize: 10
        }

        onClicked: id_control.clicked()
    }
}
