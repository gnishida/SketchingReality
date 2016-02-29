/********************************************************************************
** Form generated from reading UI file 'MainWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.5.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindowClass
{
public:
    QAction *actionExit;
    QAction *actionReconstruct;
    QAction *actionNewSketch;
    QAction *actionLoadSketch;
    QAction *actionSaveSketch;
    QAction *actionUndo;
    QWidget *centralWidget;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QMenu *menuTest;
    QMenu *menuEdit;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindowClass)
    {
        if (MainWindowClass->objectName().isEmpty())
            MainWindowClass->setObjectName(QStringLiteral("MainWindowClass"));
        MainWindowClass->resize(800, 600);
        actionExit = new QAction(MainWindowClass);
        actionExit->setObjectName(QStringLiteral("actionExit"));
        actionReconstruct = new QAction(MainWindowClass);
        actionReconstruct->setObjectName(QStringLiteral("actionReconstruct"));
        actionNewSketch = new QAction(MainWindowClass);
        actionNewSketch->setObjectName(QStringLiteral("actionNewSketch"));
        actionLoadSketch = new QAction(MainWindowClass);
        actionLoadSketch->setObjectName(QStringLiteral("actionLoadSketch"));
        actionSaveSketch = new QAction(MainWindowClass);
        actionSaveSketch->setObjectName(QStringLiteral("actionSaveSketch"));
        actionUndo = new QAction(MainWindowClass);
        actionUndo->setObjectName(QStringLiteral("actionUndo"));
        centralWidget = new QWidget(MainWindowClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        MainWindowClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindowClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 800, 21));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        menuTest = new QMenu(menuBar);
        menuTest->setObjectName(QStringLiteral("menuTest"));
        menuEdit = new QMenu(menuBar);
        menuEdit->setObjectName(QStringLiteral("menuEdit"));
        MainWindowClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindowClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        MainWindowClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindowClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindowClass->setStatusBar(statusBar);

        menuBar->addAction(menuFile->menuAction());
        menuBar->addAction(menuEdit->menuAction());
        menuBar->addAction(menuTest->menuAction());
        menuFile->addAction(actionNewSketch);
        menuFile->addAction(actionLoadSketch);
        menuFile->addAction(actionSaveSketch);
        menuFile->addSeparator();
        menuFile->addAction(actionExit);
        menuTest->addAction(actionReconstruct);
        menuEdit->addAction(actionUndo);

        retranslateUi(MainWindowClass);

        QMetaObject::connectSlotsByName(MainWindowClass);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindowClass)
    {
        MainWindowClass->setWindowTitle(QApplication::translate("MainWindowClass", "MainWindow", 0));
        actionExit->setText(QApplication::translate("MainWindowClass", "Exit", 0));
        actionReconstruct->setText(QApplication::translate("MainWindowClass", "Reconstruct", 0));
        actionNewSketch->setText(QApplication::translate("MainWindowClass", "New Sketch", 0));
        actionNewSketch->setShortcut(QApplication::translate("MainWindowClass", "Ctrl+N", 0));
        actionLoadSketch->setText(QApplication::translate("MainWindowClass", "Load Sketch", 0));
        actionLoadSketch->setShortcut(QApplication::translate("MainWindowClass", "Ctrl+O", 0));
        actionSaveSketch->setText(QApplication::translate("MainWindowClass", "Save Sketch", 0));
        actionSaveSketch->setShortcut(QApplication::translate("MainWindowClass", "Ctrl+S", 0));
        actionUndo->setText(QApplication::translate("MainWindowClass", "Undo", 0));
        actionUndo->setShortcut(QApplication::translate("MainWindowClass", "Ctrl+Z", 0));
        menuFile->setTitle(QApplication::translate("MainWindowClass", "File", 0));
        menuTest->setTitle(QApplication::translate("MainWindowClass", "Test", 0));
        menuEdit->setTitle(QApplication::translate("MainWindowClass", "Edit", 0));
    } // retranslateUi

};

namespace Ui {
    class MainWindowClass: public Ui_MainWindowClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
