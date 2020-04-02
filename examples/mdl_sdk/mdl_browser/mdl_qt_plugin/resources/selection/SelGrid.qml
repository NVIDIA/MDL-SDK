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

import "../search" as Search

RowLayout {
    id: id_control

    property real scrollBarWidth: 10
    width: parent.width
    height: parent.height

    property int elementSize: 160
    property int sideMargin: 30
    property int topMargin: 15
    property int bottomMargin: 10

    property var model: sel_mockup

    Connections {
        target: model
        // clear selection when the filter changed
        onFiltering_about_to_start: id_grid.currentIndex = -1
    }

    function clearSelection() {
        id_grid.currentIndex = -1
    }



    GridView {
        id: id_grid

        Layout.fillHeight: true
        Layout.fillWidth: true
        Layout.leftMargin: { sideMargin + ((id_control.width - 2 * sideMargin) % id_control.elementSize) / 2 }

        cellWidth: id_control.elementSize
        cellHeight: id_control.elementSize + 30

        currentIndex: -1

        header: Item {
            height: id_control.topMargin
        }
        footer: Item {
            height: id_control.bottomMargin
        }

        focus: true
        clip: true

        model: id_control.model

        delegate: SelGridItem {
            width: id_grid.cellWidth
            height: id_grid.cellHeight
            isSelected: id_grid.currentIndex == index

            onClicked: {
                if(id_grid.currentIndex == index)
                    id_grid.currentIndex = -1
                else
                    id_grid.currentIndex = index
            }

            onConfirmed: {
                vm_mdl_browser.set_result_and_close(value);
            }
        }

        ScrollBar.vertical: ScrollBar {
            implicitWidth: id_control.scrollBarWidth
        }
    }
}
