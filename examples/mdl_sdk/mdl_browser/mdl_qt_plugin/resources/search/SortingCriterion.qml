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

ItemDelegate {
    id: id_sort_name
    height: 20 // parent.height
    implicitWidth: id_name_label.implicitWidth + 15

    property bool isSelected: true
    property bool ascending: true
    property string name: "Name"

    signal selected();

    Label {
        id: id_name_label
        anchors.verticalCenter: parent.verticalCenter
        text: parent.name
        font.pointSize: 10
        color: (isSelected ? Material.accent : Material.foreground)
    }

    Label {
        anchors.verticalCenter: parent.verticalCenter
        anchors.verticalCenterOffset: -1
        anchors.left: id_name_label.right
        anchors.leftMargin: 5
        text: parent.ascending ? "↓" : "↑"
        font.pointSize: 10
        color: (isSelected ? Material.accent : Material.foreground)
    }

    onClicked: {
        // if this is currently selected, we change the order
        if(isSelected)
        {
            ascending = !ascending;
            selected();
        }
        // otherwise, we switch without changing the order
        else
        {
            selected();
        }
    }
}
