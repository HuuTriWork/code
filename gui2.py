import sys
import subprocess
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QCheckBox, QPushButton, QTextEdit,
    QHeaderView, QLabel, QTabWidget, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon

ADB_PATH = "adb\\adb.exe"
connected_devices = set()


def get_ldplayer_devices():
    result = subprocess.run([ADB_PATH, "devices"], capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()[1:]
    devices = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        device_id = parts[0]
        if device_id.startswith("emulator-"):
            devices.append(device_id)
    return devices


def auto_connect(dev):
    if dev in connected_devices:
        return
    try:
        port_num = int(dev.split("-")[-1]) + 1
        ip_port = f"127.0.0.1:{port_num}"
        subprocess.run([ADB_PATH, "connect", ip_port],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        connected_devices.add(dev)
    except Exception:
        pass


def launch_game(dev, package_name):
    subprocess.run([ADB_PATH, "-s", dev, "shell", "monkey",
                    "-p", package_name, "-c", "android.intent.category.LAUNCHER", "1"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def close_game(dev, package_name):
    subprocess.run([ADB_PATH, "-s", dev, "shell", "am", "force-stop", package_name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rise Of Kingdoms")
        self.setWindowIcon(QIcon("logo.png"))
        self.resize(250, 500)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(6, 6, 6, 6)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["#", "Emulator", "Trạng thái"])
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)

        self.table.setStyleSheet("""
            QTableWidget {
                border: 1px solid gray;
            }
            QHeaderView::section {
                border: 1px solid gray;
                background-color: #f0f0f0;
            }
            QTableWidget::item {
                border: 1px solid gray;
            }
        """)

        frame_emulator = QFrame()
        vbox_emulator = QVBoxLayout()
        vbox_emulator.setContentsMargins(0, 0, 0, 0)
        vbox_emulator.addWidget(self.table)
        frame_emulator.setLayout(vbox_emulator)
        main_layout.addWidget(frame_emulator)

        frame_logs = QFrame()
        vbox_logs = QVBoxLayout()
        vbox_logs.setContentsMargins(0, 0, 0, 0)

        lbl_logs = QLabel("Activity Logs")
        lbl_logs.setStyleSheet("""
            background-color: #f0f0f0;
            font-weight: bold;
            padding: 4px;
            border: 1px solid gray;
        """)
        vbox_logs.addWidget(lbl_logs)

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setStyleSheet("border: 1px solid gray; border-top: none;")
        vbox_logs.addWidget(self.logs)

        frame_logs.setLayout(vbox_logs)
        main_layout.addWidget(frame_logs)

        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid gray;
            }
            QTabBar::tab {
                padding: 6px;
            }
        """)

        tab_control = QWidget()
        layout_control = QVBoxLayout()

        hbox_global = QHBoxLayout()
        self.btn_open_game_global = QPushButton("Open Global")
        self.btn_close_game_global = QPushButton("Close Global")
        hbox_global.addWidget(self.btn_open_game_global)
        hbox_global.addWidget(self.btn_close_game_global)
        layout_control.addLayout(hbox_global)

        hbox_vn = QHBoxLayout()
        self.btn_open_game_vn = QPushButton("Open VN")
        self.btn_close_game_vn = QPushButton("Close VN")
        hbox_vn.addWidget(self.btn_open_game_vn)
        hbox_vn.addWidget(self.btn_close_game_vn)
        layout_control.addLayout(hbox_vn)

        tab_control.setLayout(layout_control)
        tabs.addTab(tab_control, "Game")

        frame_tabs = QFrame()
        vbox_tabs = QVBoxLayout()
        vbox_tabs.setContentsMargins(0, 0, 0, 0)
        vbox_tabs.addWidget(tabs)
        frame_tabs.setLayout(vbox_tabs)
        main_layout.addWidget(frame_tabs)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.btn_open_game_global.clicked.connect(lambda: self.launch_close_game("open", "com.lilithgame.roc.gp"))
        self.btn_close_game_global.clicked.connect(lambda: self.launch_close_game("close", "com.lilithgame.roc.gp"))
        self.btn_open_game_vn.clicked.connect(lambda: self.launch_close_game("open", "com.rok.gp.vn"))
        self.btn_close_game_vn.clicked.connect(lambda: self.launch_close_game("close", "com.rok.gp.vn"))

        self.timer = QTimer()
        self.timer.timeout.connect(self.scan_and_connect)
        self.timer.start(3000)
        self.scan_and_connect()

    def log(self, msg):
        now = time.strftime("%H:%M:%S")
        self.logs.append(f"{now} - {msg}")

    def get_selected_devices(self):
        devices = []
        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, 0)
            if widget:
                chk = widget.layout().itemAt(0).widget()
                if chk and chk.isChecked():
                    dev = self.table.item(row, 1).text()
                    devices.append(dev)
        return devices

    def launch_close_game(self, action, pkg):
        selected = self.get_selected_devices()
        if not selected:
            self.log("Chưa chọn thiết bị nào để thao tác !")
            return
        for dev in selected:
            try:
                if action == "open":
                    launch_game(dev, pkg)
                    self.log(f"Đã mở game ({pkg}) trên {dev}")
                else:
                    close_game(dev, pkg)
                    self.log(f"Đã đóng game ({pkg}) trên {dev}")
            except Exception as e:
                self.log(f"Lỗi thao tác game trên {dev}: {e}")

    def scan_and_connect(self):
        try:
            previously_selected = set(self.get_selected_devices())
            devices = get_ldplayer_devices()
            for dev in devices:
                auto_connect(dev)

            self.table.setRowCount(len(devices))
            for row, dev in enumerate(devices):
                chk = QCheckBox()
                if dev in previously_selected:
                    chk.setChecked(True)

                layout = QHBoxLayout()
                layout.addWidget(chk)
                layout.setAlignment(Qt.AlignCenter)
                layout.setContentsMargins(0, 0, 0, 0)
                w = QWidget()
                w.setLayout(layout)
                self.table.setCellWidget(row, 0, w)

                item_dev = QTableWidgetItem(dev)
                item_dev.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, 1, item_dev)

                item_status = QTableWidgetItem("🟢 Đã kết nối" if dev in connected_devices else "🔴 Lỗi")
                item_status.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, 2, item_status)

        except Exception as e:
            self.log(f"Lỗi quét thiết bị: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
