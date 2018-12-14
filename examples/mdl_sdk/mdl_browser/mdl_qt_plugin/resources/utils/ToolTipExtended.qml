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
