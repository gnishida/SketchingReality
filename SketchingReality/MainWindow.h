#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtWidgets/QMainWindow>
#include "ui_MainWindow.h"
#include "GLWidget3D.h"

class MainWindow : public QMainWindow {
	Q_OBJECT

private:
	Ui::MainWindowClass ui;
	GLWidget3D* glWidget;

public:
	MainWindow(QWidget *parent = 0);

public slots:
	void onNewSketch();
	void onLoadSketch();
	void onSaveSketch();
	void onUndo();
	void onReconstruct();
};

#endif // MAINWINDOW_H
