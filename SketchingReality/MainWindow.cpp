#include "MainWindow.h"
#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
	ui.setupUi(this);

	// メニューハンドラ
	connect(ui.actionNewSketch, SIGNAL(triggered()), this, SLOT(onNewSketch()));
	connect(ui.actionLoadSketch, SIGNAL(triggered()), this, SLOT(onLoadSketch()));
	connect(ui.actionSaveSketch, SIGNAL(triggered()), this, SLOT(onSaveSketch()));
	connect(ui.actionExit, SIGNAL(triggered()), this, SLOT(close()));
	connect(ui.actionUndo, SIGNAL(triggered()), this, SLOT(onUndo()));
	connect(ui.actionReconstruct, SIGNAL(triggered()), this, SLOT(onReconstruct()));

	glWidget = new GLWidget3D(this);
	setCentralWidget(glWidget);
}

void MainWindow::onNewSketch() {
	glWidget->clearSketch();
	glWidget->update();
}

void MainWindow::onLoadSketch() {
	QString filename = QFileDialog::getOpenFileName(this, tr("Open sketch file..."), "", tr("sketch Files (*.skt)"));
	if (filename.isEmpty()) return;
	
	glWidget->loadSketch(filename);
	glWidget->update();
}

void MainWindow::onSaveSketch() {
	QString filename = QFileDialog::getSaveFileName(this, tr("Save sketch file..."), "", tr("Sketch Files (*.skt)"));
	if (filename.isEmpty()) return;

	glWidget->saveSketch(filename);
}

void MainWindow::onUndo() {
	glWidget->undo();
	glWidget->update();
}

void MainWindow::onReconstruct() {
	glWidget->reconstruct();
}