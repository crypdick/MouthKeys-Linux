"""Main Settings window stylesheet; platform color selection is supplied by Qt."""


def window_stylesheet(preference: str, *, system_light: bool) -> str:
    light = preference == "light" or (preference == "system" and system_light)
    surface, ink, edge, side, field = (
        ("#ffffff", "#111214", "#111214", "#f2f2f4", "#ffffff")
        if light
        else ("#111214", "#ebebec", "#2c2e33", "#0c0d0f", "#0c0d0f")
    )
    return (
        f"QWidget {{ background: {surface}; color: {ink}; font-family: sans-serif;"
        "font-size: 13px; }"
        f"QWidget#sidebar {{ background: {side}; border-right: 1px solid {edge}; }}"
        f"#sidebar QLabel, #navigation {{ background: {side}; font-family: monospace; }}"
        "#brand { font-size: 20px; font-weight: 600; letter-spacing: 2px; }"
        "#placard { font-size: 10px; }"
        "#pageTitle { font-size: 24px; font-weight: 600; }"
        f"QPushButton {{ padding: 8px 12px; border: 1px solid {edge}; border-radius: 0; }}"
        f"QPushButton:hover {{ border-color: {ink}; }}"
        f"QPushButton:pressed {{ background: {ink}; color: {surface}; }}"
        "QPushButton:disabled { color: #777; }"
        f"QLineEdit, QTextEdit, QListWidget, QComboBox, QSpinBox {{ background: {field};"
        f"border: 1px solid {edge}; padding: 8px; selection-background-color: {ink};"
        f"selection-color: {surface}; }}"
        "QListWidget#navigation { border: 0; padding: 0; }"
        "QListWidget::item { padding: 12px 8px; }"
        f"QListWidget::item:selected {{ background: {ink}; color: {surface}; }}"
    )
