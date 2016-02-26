#pragma once

#include "glew.h"
#include <QGLWidget>
#include <QMouseEvent>
#include <QKeyEvent>
#include "Camera.h"
#include "ShadowMapping.h"
#include "RenderManager.h"
#include <vector>
#include <QPen>
#include "SketchGraph.h"

class MainWindow;

class GLWidget3D : public QGLWidget {
public:
	MainWindow* mainWin;
	Camera camera;
	glm::vec3 light_dir;
	glm::mat4 light_mvpMatrix;
	RenderManager renderManager;
	QPoint lastPos;
	bool ctrlPressed;
	bool shiftPressed;
	bool altPressed;
	
	QPoint lastPoint;
	QImage sketch;
	std::vector<std::vector<glm::vec2> > strokes;
	std::vector<glm::vec2> current_stroke;

	sketch::SketchGraph sketchGraph;

public:
	GLWidget3D(MainWindow *parent);
	void render();
	void clearSketch();
	void undo();
	void loadSketch(const QString& filename);
	void saveSketch(const QString& filename);
	void drawScene();
	void drawLine(const QPoint& startPoint, const QPoint& endPoint);
	void computeVanishingPoints(std::vector<sketch::VanishingPoint>& pv);
	void reconstruct();
	void resizeSketch(int width, int height);

	void keyPressEvent(QKeyEvent* e);
	void keyReleaseEvent(QKeyEvent* e);

protected:
	void mousePressEvent(QMouseEvent* e);
	void mouseMoveEvent(QMouseEvent* e);
	void mouseReleaseEvent(QMouseEvent* e);
	void wheelEvent(QWheelEvent* e);
	void initializeGL();
	void resizeGL(int width, int height);
	void paintEvent(QPaintEvent* e);
};

