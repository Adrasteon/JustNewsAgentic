import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox
)
from PyQt5.QtCore import Qt
import subprocess
import threading
import requests

class DashboardGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dashboard Agent GUI")
        self.setGeometry(100, 100, 800, 600)

        # Apply dark theme stylesheet
        dark_grey = "#232323"
        white = "#ffffff"
        self.setStyleSheet(f"""
            QMainWindow {{ background: {dark_grey}; color: {white}; }}
            QWidget {{ background: {dark_grey}; color: {white}; }}
            QTabWidget::pane {{ background: {dark_grey}; }}
            QTabBar::tab {{ background: {dark_grey}; color: {white}; padding: 8px; }}
            QTabBar::tab:selected {{ background: #333333; color: {white}; }}
            QLabel {{ color: {white}; }}
            QPushButton {{ background: #444444; color: {white}; border: 1px solid #666; border-radius: 4px; padding: 4px 12px; }}
            QPushButton:disabled {{ background: #222; color: #888; }}
        """)

    # Create the tab widget and add all tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self.create_monitoring_tab(), "Monitoring")
        self.tabs.addTab(self.create_analysis_tab(), "Analysis")
        self.tabs.addTab(self.create_services_tab(), "Services")
        self.tabs.addTab(self.create_web_crawl_tab(), "Web Crawl")
        self.tabs.addTab(self.create_settings_tab(), "Settings")
    def create_web_crawl_tab(self):
        from PyQt5.QtWidgets import QCheckBox
        tab = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Web Crawl Targets")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # List of URLs with checkboxes
        self.crawl_urls = ["https://www.bbc.com/news", "https://www.cnn.com/world", "https://www.reuters.com/news/world"]
        self.crawl_url_checkboxes = []
        self.crawl_url_layout = QVBoxLayout()
        for url in self.crawl_urls:
            row = QHBoxLayout()
            cb = QCheckBox(url)
            cb.setChecked(True)
            cb.setStyleSheet("color: #fff; padding: 4px 8px;")
            cb.stateChanged.connect(lambda state, box=cb: self.update_crawl_url_style(box))
            row.addWidget(cb)
            self.crawl_url_checkboxes.append(cb)
            self.crawl_url_layout.addLayout(row)
        layout.addLayout(self.crawl_url_layout)

        # Add new URL row
        add_row = QHBoxLayout()
        self.new_url_input = QLabel("[Click + to add new URL]")
        self.new_url_input.setStyleSheet("color: #888; font-style: italic;")
        add_btn = QPushButton("+")
        add_btn.setFixedWidth(32)
        add_btn.setStyleSheet("background: #2e7d32; color: #fff; font-weight: bold; font-size: 18px;")
        add_btn.clicked.connect(self.add_new_crawl_url)
        add_row.addWidget(self.new_url_input)
        add_row.addWidget(add_btn)
        layout.addLayout(add_row)

        # Start/Stop Crawl button
        self.crawl_active = False
        self.crawl_toggle_btn = QPushButton("Start Crawl")
        self.crawl_toggle_btn.setStyleSheet("background: #1976d2; color: #fff; font-weight: bold; padding: 8px 24px; border-radius: 6px;")
        self.crawl_toggle_btn.setFixedWidth(140)
        self.crawl_toggle_btn.clicked.connect(self.toggle_crawl)
        layout.addWidget(self.crawl_toggle_btn)

        # Status label
        self.crawl_status_label = QLabel("")
        self.crawl_status_label.setStyleSheet("color: #fff; margin-top: 8px;")
        layout.addWidget(self.crawl_status_label)

        # Spacer
        layout.addStretch()

        tab.setLayout(layout)
        return tab

    def update_crawl_url_style(self, cb):
        if cb.isChecked():
            cb.setStyleSheet("color: #fff; padding: 4px 8px;")
        else:
            cb.setStyleSheet("color: #888; padding: 4px 8px;")

    def add_new_crawl_url(self):
        from PyQt5.QtWidgets import QInputDialog, QCheckBox, QHBoxLayout
        url, ok = QInputDialog.getText(self, "Add Crawl Target", "Enter new target URL:")
        if ok and url:
            row = QHBoxLayout()
            cb = QCheckBox(url)
            cb.setChecked(True)
            cb.setStyleSheet("color: #fff; padding: 4px 8px;")
            cb.stateChanged.connect(lambda state, box=cb: self.update_crawl_url_style(box))
            row.addWidget(cb)
            self.crawl_url_checkboxes.append(cb)
            self.crawl_url_layout.addLayout(row)
            self.crawl_urls.append(url)

    def toggle_crawl(self):
        self.crawl_active = not self.crawl_active
        if self.crawl_active:
            self.crawl_toggle_btn.setText("Stop Crawl")
            self.crawl_toggle_btn.setStyleSheet("background: #c62828; color: #fff; font-weight: bold; padding: 8px 24px; border-radius: 6px;")
            self.crawl_status_label.setText("Crawling started for selected targets.")
        else:
            self.crawl_toggle_btn.setText("Start Crawl")
            self.crawl_toggle_btn.setStyleSheet("background: #1976d2; color: #fff; font-weight: bold; padding: 8px 24px; border-radius: 6px;")
            self.crawl_status_label.setText("Crawling stopped.")


    def create_services_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # --- Top Spacer (one line) ---
        layout.addSpacing(10)

        # --- Start/Stop All Services Row ---
        all_row = QHBoxLayout()
        all_label = QLabel("All Agents")
        all_label.setStyleSheet("font-weight: bold;")
        all_status = QLabel("")
        all_start_btn = QPushButton("Start All")
        all_stop_btn = QPushButton("Stop All")
        all_start_btn.setFixedWidth(60)  # Half the previous default width (was default, now 60)
        all_stop_btn.setFixedWidth(60)
        all_start_btn.clicked.connect(self.start_all_agents)
        all_stop_btn.clicked.connect(self.stop_all_agents)
        all_row.addWidget(all_label)
        all_row.addWidget(all_status)
        all_row.addWidget(all_start_btn)
        all_row.addWidget(all_stop_btn)
        layout.addLayout(all_row)

        # --- Extra Spacer between All Agents and rest ---
        layout.addSpacing(20)

        self.agent_info = [
            ("MCP Bus", 8000),
            ("Chief Editor Agent", 8001),
            ("Scout Agent", 8002),
            ("Fact Checker Agent", 8003),
            ("Analyst Agent", 8004),
            ("Synthesizer Agent", 8005),
            ("Critic Agent", 8006),
            ("Memory Agent", 8007),
            ("Reasoning Agent", 8008),
            ("NewsReader Agent", 8009),
        ]

        self.agent_buttons = {}
        self.all_status_label = all_status

        for name, port in self.agent_info:
            row = QHBoxLayout()
            label = QLabel(f"{name} (port {port})")
            status_label = QLabel("Checking...")
            start_btn = QPushButton("Start")
            stop_btn = QPushButton("Stop")
            start_btn.setFixedWidth(60)
            stop_btn.setFixedWidth(60)
            start_btn.clicked.connect(lambda _, n=name: self.start_agent(n))
            stop_btn.clicked.connect(lambda _, n=name: self.stop_agent(n))
            row.addWidget(label)
            row.addWidget(status_label)
            row.addWidget(start_btn)
            row.addWidget(stop_btn)
            layout.addLayout(row)
            self.agent_buttons[name] = (status_label, start_btn, stop_btn, port)

        # Initial status check
        threading.Thread(target=self.update_all_status, daemon=True).start()

        tab.setLayout(layout)
        return tab

    def start_all_agents(self):
        self.all_status_label.setText("Starting...")
        self.all_status_label.setStyleSheet("color: orange;")
        for name in self.agent_buttons:
            status_label, start_btn, stop_btn, _ = self.agent_buttons[name]
            status_label.setText("Starting...")
            status_label.setStyleSheet("color: orange;")
            start_btn.setEnabled(False)
            stop_btn.setEnabled(False)
        threading.Thread(target=self._start_all_agents_thread, daemon=True).start()

    def _start_all_agents_thread(self):
        subprocess.call(["/bin/bash", "./start_services_daemon.sh"])
        self.update_all_status()
        self.all_status_label.setText("")

    def stop_all_agents(self):
        self.all_status_label.setText("Stopping...")
        self.all_status_label.setStyleSheet("color: orange;")
        for name in self.agent_buttons:
            status_label, start_btn, stop_btn, _ = self.agent_buttons[name]
            status_label.setText("Stopping...")
            status_label.setStyleSheet("color: orange;")
            start_btn.setEnabled(False)
            stop_btn.setEnabled(False)
        threading.Thread(target=self._stop_all_agents_thread, daemon=True).start()

    def _stop_all_agents_thread(self):
        subprocess.call(["/bin/bash", "./stop_services.sh"])
        self.update_all_status()
        self.all_status_label.setText("")

    def update_all_status(self):
        for name, (_, _, _, port) in self.agent_buttons.items():
            self.update_agent_status(name, port)

    def update_agent_status(self, name, port):
        status_label, start_btn, stop_btn, _ = self.agent_buttons[name]
        try:
            if port == 8000:
                # MCP Bus
                resp = requests.get(f"http://localhost:{port}/agents", timeout=1)
            else:
                resp = requests.get(f"http://localhost:{port}/health", timeout=1)
            if resp.status_code == 200:
                status_label.setText("Running")
                status_label.setStyleSheet("color: green;")
                start_btn.setEnabled(False)
                stop_btn.setEnabled(True)
            else:
                status_label.setText("Stopped")
                status_label.setStyleSheet("color: red;")
                start_btn.setEnabled(True)
                stop_btn.setEnabled(False)
        except Exception:
            status_label.setText("Stopped")
            status_label.setStyleSheet("color: red;")
            start_btn.setEnabled(True)
            stop_btn.setEnabled(False)

    def start_agent(self, name):
        # Start the agent using the start_services_daemon.sh script
        msg = QMessageBox()
        msg.setWindowTitle("Start Agent")
        msg.setText(f"Starting {name}... (this may take a few seconds)")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.show()
        threading.Thread(target=self._start_agent_thread, args=(name,), daemon=True).start()

    def _start_agent_thread(self, name):
        # Map display name to script argument
        agent_map = {
            "MCP Bus": "mcp_bus",
            "Chief Editor Agent": "chief_editor",
            "Scout Agent": "scout",
            "Fact Checker Agent": "fact_checker",
            "Analyst Agent": "analyst",
            "Synthesizer Agent": "synthesizer",
            "Critic Agent": "critic",
            "Memory Agent": "memory",
            "Reasoning Agent": "reasoning",
            "NewsReader Agent": "newsreader",
        }
        script_arg = agent_map.get(name)
        if script_arg:
            subprocess.call(["/bin/bash", "./start_services_daemon.sh", script_arg])
        # Wait a bit and update status
        self.update_agent_status(name, self.agent_buttons[name][3])

    def stop_agent(self, name):
        # Stop the agent using the stop_services.sh script
        msg = QMessageBox()
        msg.setWindowTitle("Stop Agent")
        msg.setText(f"Stopping {name}... (this may take a few seconds)")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.show()
        threading.Thread(target=self._stop_agent_thread, args=(name,), daemon=True).start()

    def _stop_agent_thread(self, name):
        agent_map = {
            "MCP Bus": "mcp_bus",
            "Chief Editor Agent": "chief_editor",
            "Scout Agent": "scout",
            "Fact Checker Agent": "fact_checker",
            "Analyst Agent": "analyst",
            "Synthesizer Agent": "synthesizer",
            "Critic Agent": "critic",
            "Memory Agent": "memory",
            "Reasoning Agent": "reasoning",
            "NewsReader Agent": "newsreader",
            "Dashboard Agent": "dashboard",
        }
        script_arg = agent_map.get(name)
        if script_arg:
            subprocess.call(["/bin/bash", "./stop_services.sh", script_arg])
        # Wait a bit and update status
        self.update_agent_status(name, self.agent_buttons[name][3])

    def create_monitoring_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Example content for Monitoring tab
        layout.addWidget(QLabel("Real-time system and agent activity will be displayed here."))

        tab.setLayout(layout)
        return tab

    def create_analysis_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Example content for Analysis tab
        layout.addWidget(QLabel("Tools for analyzing articles will be displayed here."))

        tab.setLayout(layout)
        return tab

    def create_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Example content for Settings tab
        layout.addWidget(QLabel("Configuration options for the Dashboard Agent will be displayed here."))

        tab.setLayout(layout)
        return tab

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DashboardGUI()
    gui.show()
    sys.exit(app.exec_())
