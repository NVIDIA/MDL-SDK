import QtQuick 2.10
import QtQuick.Controls 2.3
import QtQuick.Controls.Material 2.3
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0


Item {
    id: id_control
    width: parent.width
    height: parent.height

    Rectangle {
        height: 100
        anchors.fill: parent
        color: Material.background

        layer.enabled: true
        layer.effect: DropShadow {
            horizontalOffset: 0
            verticalOffset: 0
            radius: 12.0
            samples: 16
            color: "#BB000000"
        }
    }
}
