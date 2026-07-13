DARK_STYLESHEET = """
QMainWindow {
    background-color: #1e1e1e;
}

QWidget {
    background-color: #2d2d2d;
    color: #d4d4d4;
    font-family: "Segoe UI";
    font-size: 12px;
}

QListWidget {
    background-color: #252526;
    border: none;
    padding: 8px 4px;
}

QListWidget::item {
    padding: 12px 16px;
    border-radius: 6px;
    margin: 2px 8px;
}

QListWidget::item:selected {
    background-color: #094771;
    color: #ffffff;
}

QListWidget::item:hover {
    background-color: #3c3c3c;
}

QGroupBox {
    font-weight: bold;
    font-size: 13px;
    border: 1px solid #3c3c3c;
    border-radius: 8px;
    margin-top: 16px;
    padding-top: 20px;
    background-color: #2d2d2d;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 8px;
    color: #569cd6;
}

QLabel {
    color: #d4d4d4;
}

QLineEdit {
    background-color: #3c3c3c;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 6px 10px;
    color: #d4d4d4;
    selection-background-color: #094771;
}

QLineEdit:focus {
    border: 1px solid #569cd6;
}

QLineEdit:read-only {
    background-color: #333333;
    color: #888888;
}

QSpinBox, QDoubleSpinBox {
    background-color: #3c3c3c;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 6px 10px;
    color: #d4d4d4;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #569cd6;
}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #4c4c4c;
    width: 20px;
    border: none;
}

QPushButton {
    background-color: #0e639c;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 10px 24px;
    font-weight: bold;
    font-size: 13px;
}

QPushButton:hover {
    background-color: #1173b5;
}

QPushButton:pressed {
    background-color: #094771;
}

QPushButton:disabled {
    background-color: #3c3c3c;
    color: #666666;
}

QProgressBar {
    background-color: #3c3c3c;
    border: 1px solid #555555;
    border-radius: 6px;
    height: 24px;
    text-align: center;
    color: #d4d4d4;
}

QProgressBar::chunk {
    background-color: #0e639c;
    border-radius: 5px;
}

QTextBrowser {
    background-color: #1e1e1e;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    color: #d4d4d4;
}

QTableWidget {
    background-color: #2d2d2d;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    gridline-color: #3c3c3c;
    selection-background-color: #094771;
}

QTableWidget::item {
    padding: 6px 8px;
}

QHeaderView::section {
    background-color: #3c3c3c;
    color: #d4d4d4;
    border: none;
    padding: 8px;
    font-weight: bold;
}

QScrollArea {
    border: none;
}

QScrollBar:vertical {
    background-color: #2d2d2d;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background-color: #555555;
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #777777;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QFormLayout {
    spacing: 8px;
}

QComboBox {
    background-color: #3c3c3c;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 6px 10px;
    color: #d4d4d4;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #3c3c3c;
    border: 1px solid #555555;
    selection-background-color: #094771;
    color: #d4d4d4;
}
"""
