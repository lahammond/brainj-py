# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:16:58 2023

"""
__author__    = 'Luke Hammond <lh2881@columbia.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2022 by Luke Hammond'
__webpage__   = 'http://cellularimaging.org'
__download__  = 'http://www.github.com/lahmmond/BrainJ-Python'

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                             QPushButton, QCheckBox, QLabel, QLineEdit, 
                             QMessageBox, QTextEdit, QWidget, QFileDialog, 
                             QGridLayout,QHBoxLayout, QGroupBox, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSlot, QTime, QThread, pyqtSignal
from PyQt5.QtGui import QTextCursor

import pickle
import logging
from datetime import datetime

sys.path.append('D:/Dropbox/Github/BrainJ-Python/')

class Worker(QThread):
    log_generated = pyqtSignal(str)
    task_done = pyqtSignal(str) 

    def __init__(self, directory, channel_options, integers, settings, locations, logger):
        super().__init__()
        self.directory = directory
        self.channel_options = channel_options
        self.integers = integers
        self.settings = settings
        self.locations = locations
        self.logger = logger

    def run(self):
        try:
            from BrainJ.Environment import imgan
            
            log_path = self.directory +'BrainJ_Log_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.log'
            self.logger.set_log_file(log_path)  # Note: we're not passing a log_display, because it's not thread-safe

            self.logger.log(f"Processing directory: {self.directory}, channel_options: {self.channel_options}, integers: {self.integers}")
    
            #Print out relevant settings:
            self.logger.log(f"Analyze channels (C1, C2, C3, C4) set to: {self.settings.c1_cell_analysis} {self.settings.c2_cell_analysis} {self.settings.c3_cell_analysis} {self.settings.c4_cell_analysis}")
            self.logger.log(f"Measure channels (C1, C2, C3, C4) set to: {self.settings.c1_measure} {self.settings.c2_measure} {self.settings.c3_measure} {self.settings.c4_measure}")
    
            #1: restore and detect cells
            rawcells = imgan.cell_detection(self.settings, self.locations, self.logger)
    
            #2a: transform cells - use for normal processing - 2b or 2c are for reprocessing.
            transformedcells = imgan.transform_cells_V2(rawcells, self.settings, self.locations, self.logger)
    
            #Annotate cells and create tables
            imgan.annotate_all_cells(transformedcells, self.settings, self.locations, self.logger)
    
            self.task_done.emit("BrainJ processing completed.")

        
        except Exception as e:
            self.log_generated.emit(str(e))
            self.task_done.emit("An error occurred.")


class Logger:


    def __init__(self, log_file_path=None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.handler = None
        if log_file_path is not None:
            self.set_log_file(log_file_path)
        # redirect stdout to logger
        sys.stdout = self
        self.log_display = QTextEdit()

    def set_log_file(self, log_file_path):
        if self.handler is not None:
            self.logger.removeHandler(self.handler)
        self.handler = logging.FileHandler(log_file_path)
        self.handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(self.handler)

    def log(self, msg):
        msg = msg.replace('\n', ' ')  # replace newlines with space
        self.logger.debug(msg)
        
    def info(self, msg):
        self.log(msg)

    # override stdout's write method
    def write(self, text):
        self.log(text)

    # to handle the '\n' that print adds
    def flush(self):
        pass

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        widget = QWidget()
        layout = QVBoxLayout() 
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.logger = Logger()
        #self.logger.log_generated.connect(self.update_log_display)  # Connect the signal


        self.brainjdir_label = QLabel("No BrainJ directory selected.")
        self.brainjdir_button = QPushButton("Select BrainJ directory")
        self.brainjdir_button.clicked.connect(self.get_brainjdir)
        
        self.directory_label = QLabel("No directory selected.")
        self.directory_button = QPushButton("Select Directory")
        self.directory_button.clicked.connect(self.get_directories)
        


        options_group = QGroupBox("Cell Analysis Options")
        options_layout = QVBoxLayout()
        options_group.setLayout(options_layout)
        self.analyze1 = QCheckBox("Analyze Channel 1")
        self.analyze1.setChecked(True) 
        self.analyze2 = QCheckBox("Analyze Channel 2")
        self.analyze2.setChecked(True) 
        self.analyze3 = QCheckBox("Analyze Channel 3")
        self.analyze3.setChecked(True) 
        self.analyze4 = QCheckBox("Analyze Channel 4")
        self.analyze4.setChecked(True) 
        options_layout.addWidget(self.analyze1)
        options_layout.addWidget(self.analyze2)
        options_layout.addWidget(self.analyze3)
        options_layout.addWidget(self.analyze4)

        options_group2 = QGroupBox("Other Options")
        options_layout2 = QVBoxLayout()
        options_group2.setLayout(options_layout2)
        self.save_intermediate = QCheckBox("Save intermediate data")
        self.save_intermediate.setChecked(True) 

        options_layout2.addWidget(self.save_intermediate)

        self.integer_label = QLabel("Patches for processing (increase patches if processing failes):")
        self.integer_input = QLineEdit("2,2")

        run_cancel_layout = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_function)
        run_cancel_layout.addWidget(self.run_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.close)
        run_cancel_layout.addWidget(self.cancel_button)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        

        layout.addWidget(self.brainjdir_label)
        layout.addWidget(self.brainjdir_button)
        layout.addWidget(self.directory_label)
        layout.addWidget(self.directory_button)
        layout.addWidget(options_group)
        layout.addWidget(options_group2)
        layout.addWidget(self.integer_label)
        layout.addWidget(self.integer_input)
        layout.addLayout(run_cancel_layout)
        layout.addWidget(self.progress)
        layout.addWidget(self.logger.log_display)
        
        try:
            with open('last_dir.pickle', 'rb') as f:
                last_dir = pickle.load(f)
                self.directory_label.setText(f"Selected directory: {last_dir}")
        except (FileNotFoundError, EOFError, pickle.PickleError):
            pass  # If we can't load the last directory path, just ignore the error
            
        try:
            with open('last_dir2.pickle', 'rb') as f:
                last_dir2 = pickle.load(f)
                self.brainjdir_label.setText(f"Select BrainJ directory: {last_dir2}")
        except (FileNotFoundError, EOFError, pickle.PickleError):
            pass  # If we can't load the last directory2 path, just ignore the error



    @pyqtSlot()
    def get_directories(self):
        directory = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if directory:
            self.directory_label.setText(f"Selected directory: {directory}")

            # Save the selected directory path
            with open('last_dir.pickle', 'wb') as f:
                pickle.dump(directory, f)
    
    @pyqtSlot()
    def get_brainjdir(self):
        brainjdir = QFileDialog.getExistingDirectory(self, 'Select BrainJ directory')
        if brainjdir:
            self.brainjdir_label.setText(f"Select BrainJ directory: {brainjdir}")

            # Save the selected directory2 path
            with open('last_dir2.pickle', 'wb') as f:
                pickle.dump(brainjdir, f)

    @pyqtSlot()
    def run_function(self):
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Set to busy mode
        
        #import QLEAN modules
        from BrainJ.Environment import main, imgan
        main.check_gpu()
        
        
        directory = self.directory_label.text().replace("Selected directory: ", "")
        if directory == "No directory selected.":
            QMessageBox.critical(self, "Error", "No directory selected.")
            self.progress.setVisible(False)
            return

        try:
            integers = list(map(int, self.integer_input.text().split(',')))
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid integers entered.")
            self.progress.setVisible(False)
            return
        
        directory =  directory + "/"
        #Load in experiment parameters and analysis settings   
        settings, locations = main.initialize_brainJ(directory)
        
        channel_options = [self.analyze1.isChecked(), self.analyze2.isChecked(), self.analyze3.isChecked(),self.analyze4.isChecked()]
        other_options = [self.save_intermediate.isChecked()]
        
        self.worker = Worker(directory, channel_options, integers, settings, locations, self.logger)
        self.worker.log_generated.connect(self.update_log_display)  # connect here
    
        self.worker.task_done.connect(self.on_task_done)
        self.worker.start()  # This starts the task in a new thread
        
    @pyqtSlot(str)
    def update_log_display(self, message):
        self.logger.log_display.append(message)
        self.logger.log_display.moveCursor(QTextCursor.End)


    @pyqtSlot(str)
    def on_task_done(self, message):
        self.logger.log(message)
        self.progress.setValue(self.progress.maximum())
        self.progress.setVisible(False)
        
    
app = QApplication([])
window = MainWindow()
window.setWindowTitle('BrainJ')
window.setGeometry(100, 100, 1200, 800)  
window.show()
sys.exit(app.exec_())