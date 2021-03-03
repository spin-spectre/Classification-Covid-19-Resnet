# -*- coding: utf-8 -*-
import os
import sys

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from test_one import test


class ListViewDemo(QMainWindow):
    def __init__(self, parent=None):
        super(ListViewDemo, self).__init__(parent)
        self.imgName = []
        self.resize(500, 500)
        HLayout = QHBoxLayout()
        VLayout = QVBoxLayout()
        self.lab1 = QLabel()
        self.lab1.setPixmap(QPixmap("head.png"))
        HLayout.addWidget(self.lab1)
        self.listView = QListView()
        self.listView.setContextMenuPolicy(Qt.CustomContextMenu)  # 右键菜单
        self.listView.customContextMenuRequested[QtCore.QPoint].connect(self.rightMenuShow)
        self.listView.resize(300, 300)
        self.fileName = 'hh'
        self.selectbtn = QPushButton("选择图片")
        self.selectbtn.setFixedWidth(200)
        self.btnOK = QPushButton("开始检测")
        groupBox = QGroupBox("是否使用GPU")
        # self.checkBox1 = QCheckBox("&Yes")
        # self.checkBox1.setChecked(False)
        # self.checkBox1.stateChanged.connect(lambda: self.btnstate(self.checkBox1))

        layout = QHBoxLayout()  # 复选框单独用了一个水平布局
        # layout.addWidget(self.checkBox1)
        groupBox.setLayout(layout)

        VLayout.addWidget(self.btnOK)
        VLayout.addWidget(self.selectbtn)
        # VLayout.addWidget(groupBox)
        VLayout.addWidget(self.listView)

        # bar = self.menuBar()
        # file = bar.addMenu("File")
        # edit = bar.addMenu("Edit")
        # file.addAction("Open")
        # file.addAction("Save")
        # file.addAction("Close")
        self.listView.setContextMenuPolicy(Qt.CustomContextMenu)
        self.listView.customContextMenuRequested[QtCore.QPoint].connect(self.rightMenuShow)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        HLayout.addLayout(VLayout)
        main_frame = QWidget()
        main_frame.setLayout(HLayout)

        self.setWindowTitle("新冠检测系统")
        self.selectbtn.clicked.connect(self.openimage)
        self.listView.doubleClicked.connect(self.clicked)
        self.listView.clicked.connect(self.clicked)
        self.btnOK.clicked.connect(self.processimage)
        self.setCentralWidget(main_frame)

    def rightMenuShow(self):
        rightMenu = QtWidgets.QMenu(self.listView)
        removeAction = QtWidgets.QAction(u"Delete", self,
                                         triggered=self.removeimage)  # triggered 为右键菜单点击后的激活事件。这里slef.close调用的是系统自带的关闭事件。
        rightMenu.addAction(removeAction)
        rightMenu.exec_(QtGui.QCursor.pos())
        # rightMenu.resize(self,300)

    def processTrigger(self, q):
        if (q.text() == "show"):
            self.statusBar.showMessage(q.text() + " 菜单选项被点击了", 5000)

    def btnstate(self, btn):
        chk1Status = self.checkBox1.isChecked()
        print(chk1Status)
        if chk1Status:
            QMessageBox.information(self, "Tips", "使用GPU!", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        else:
            QMessageBox.information(self, "Tips", "不使用GPU!")

    def clicked(self, qModelIndex):
        # QMessageBox.information(self, "QListView", "你选择了: "+ imgName[qModelIndex.row()])
        global path
        self.lab1.setPixmap(QPixmap(imgName[qModelIndex.row()]))
        path = imgName[qModelIndex.row()]

    def openimage(self):
        global imgName
        # imgName = QtWidgets.QFileDialog.getOpenFileNames(self, "多文件选择", "/", "所有文件 (*);;文本文件 (*.txt)")
        directory1 = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        self.fileName = directory1
        imgName = self.all_path(directory1)
        # for img in directory1:
        #     imgName.append(img)
        print(imgName)
        self.lab1.setPixmap(QPixmap(imgName[0]))
        slm = QStringListModel()
        slm.setStringList(imgName)
        self.listView.setModel(slm)

    def removeimage(self):
        selected = self.listView.selectedIndexes()
        itemmodel = self.listView.model()
        for i in selected:
            itemmodel.removeRow(i.row())

    def processimage(self):
        if self.fileName == 'hh':
            return
        res = test(self.fileName)
        print(res)
        # QMessageBox.setFixedHeight(self,1000)
        # QMessageBox.setFixedWidth(self,1000)
        # QMessageBox.resize(self,2000,3000)
        if res == 0:
            QMessageBox.information(self, "检测结果", "检测的结果为" + "Covid-19")
        elif res == 1:
            QMessageBox.information(self, "检测结果", "检测的结果为" + "Non-infected")
        elif res == 2:
            QMessageBox.information(self, "检测结果", "检测的结果为" + "Cap")

    def all_path(self, dirname):
        result = []
        for maindir, subdir, file_name_list in os.walk(dirname):
            for filename in file_name_list:
                apath = os.path.join(maindir, filename)
                result.append(apath)
        return result


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ListViewDemo()
    win.show()
    sys.exit(app.exec_())
