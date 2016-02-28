#include <iostream>
#include "GLWidget3D.h"
#include "MainWindow.h"
#include <GL/GLU.h>
#include <QTimer>
#include <glm/gtc/matrix_transform.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <QDir>
#include <QMessageBox>
#include <QTextStream>
#include "Utils.h"

#ifndef M_PI
#define M_PI	3.14159265359
#endif

GLWidget3D::GLWidget3D(MainWindow* mainWin) : QGLWidget(QGLFormat(QGL::SampleBuffers), (QWidget*)mainWin) {
	this->mainWin = mainWin;
	ctrlPressed = false;
	shiftPressed = false;
	altPressed = false;

	// This is necessary to prevent the screen overdrawn by OpenGL
	setAutoFillBackground(false);

	// 光源位置をセット
	// ShadowMappingは平行光源を使っている。この位置から原点方向を平行光源の方向とする。
	light_dir = glm::normalize(glm::vec3(-4, -5, -8));
	//light_dir = glm::normalize(glm::vec3(-1, -3, -2));

	// シャドウマップ用のmodel/view/projection行列を作成
	glm::mat4 light_pMatrix = glm::ortho<float>(-100, 100, -100, 100, 0.1, 200);
	glm::mat4 light_mvMatrix = glm::lookAt(-light_dir * 50.0f, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
	light_mvpMatrix = light_pMatrix * light_mvMatrix;
}

void GLWidget3D::render() {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// PASS 1: Render to texture
	glUseProgram(renderManager.programs["pass1"]);

	glBindFramebuffer(GL_FRAMEBUFFER, renderManager.fragDataFB);
	glClearColor(0.95, 0.95, 0.95, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderManager.fragDataTex[0], 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, renderManager.fragDataTex[1], 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, renderManager.fragDataTex[2], 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, renderManager.fragDataTex[3], 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, renderManager.fragDepthTex, 0);

	// Set the list of draw buffers.
	GLenum DrawBuffers[4] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 };
	glDrawBuffers(4, DrawBuffers); // "3" is the size of DrawBuffers
	// Always check that our framebuffer is ok
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		printf("+ERROR: GL_FRAMEBUFFER_COMPLETE false\n");
		exit(0);
	}

	glUniformMatrix4fv(glGetUniformLocation(renderManager.programs["pass1"], "mvpMatrix"), 1, false, &camera.mvpMatrix[0][0]);
	glUniform3f(glGetUniformLocation(renderManager.programs["pass1"], "lightDir"), light_dir.x, light_dir.y, light_dir.z);
	glUniformMatrix4fv(glGetUniformLocation(renderManager.programs["pass1"], "light_mvpMatrix"), 1, false, &light_mvpMatrix[0][0]);

	glUniform1i(glGetUniformLocation(renderManager.programs["pass1"], "shadowMap"), 6);
	glActiveTexture(GL_TEXTURE6);
	glBindTexture(GL_TEXTURE_2D, renderManager.shadow.textureDepth);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	drawScene();

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// PASS 2: Create AO
	if (renderManager.renderingMode == RenderManager::RENDERING_MODE_SSAO) {
		glUseProgram(renderManager.programs["ssao"]);
		glBindFramebuffer(GL_FRAMEBUFFER, renderManager.fragDataFB_AO);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderManager.fragAOTex, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, renderManager.fragDepthTex_AO, 0);
		GLenum DrawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
		glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers

		glClearColor(1, 1, 1, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Always check that our framebuffer is ok
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
			printf("++ERROR: GL_FRAMEBUFFER_COMPLETE false\n");
			exit(0);
		}

		glDisable(GL_DEPTH_TEST);
		glDepthFunc(GL_ALWAYS);

		glUniform2f(glGetUniformLocation(renderManager.programs["ssao"], "pixelSize"), 2.0f / this->width(), 2.0f / this->height());

		glUniform1i(glGetUniformLocation(renderManager.programs["ssao"], "tex0"), 1);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, renderManager.fragDataTex[0]);

		glUniform1i(glGetUniformLocation(renderManager.programs["ssao"], "tex1"), 2);
		glActiveTexture(GL_TEXTURE2);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, renderManager.fragDataTex[1]);

		glUniform1i(glGetUniformLocation(renderManager.programs["ssao"], "tex2"), 3);
		glActiveTexture(GL_TEXTURE3);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, renderManager.fragDataTex[2]);

		glUniform1i(glGetUniformLocation(renderManager.programs["ssao"], "depthTex"), 8);
		glActiveTexture(GL_TEXTURE8);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, renderManager.fragDepthTex);

		glUniform1i(glGetUniformLocation(renderManager.programs["ssao"], "noiseTex"), 7);
		glActiveTexture(GL_TEXTURE7);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, renderManager.fragNoiseTex);

		{
			glUniformMatrix4fv(glGetUniformLocation(renderManager.programs["ssao"], "mvpMatrix"), 1, false, &camera.mvpMatrix[0][0]);
			glUniformMatrix4fv(glGetUniformLocation(renderManager.programs["ssao"], "pMatrix"), 1, false, &camera.pMatrix[0][0]);
		}

		glUniform1i(glGetUniformLocation(renderManager.programs["ssao"], "uKernelSize"), renderManager.uKernelSize);
		glUniform3fv(glGetUniformLocation(renderManager.programs["ssao"], "uKernelOffsets"), renderManager.uKernelOffsets.size(), (const GLfloat*)renderManager.uKernelOffsets.data());

		glUniform1f(glGetUniformLocation(renderManager.programs["ssao"], "uPower"), renderManager.uPower);
		glUniform1f(glGetUniformLocation(renderManager.programs["ssao"], "uRadius"), renderManager.uRadius);

		glBindVertexArray(renderManager.secondPassVAO);

		glDrawArrays(GL_QUADS, 0, 4);
		glBindVertexArray(0);
		glDepthFunc(GL_LEQUAL);
	}
	else if (renderManager.renderingMode == RenderManager::RENDERING_MODE_LINE || renderManager.renderingMode == RenderManager::RENDERING_MODE_HATCHING) {
		glUseProgram(renderManager.programs["line"]);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glClearColor(1, 1, 1, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glDisable(GL_DEPTH_TEST);
		glDepthFunc(GL_ALWAYS);

		glUniform2f(glGetUniformLocation(renderManager.programs["line"], "pixelSize"), 1.0f / this->width(), 1.0f / this->height());
		glUniformMatrix4fv(glGetUniformLocation(renderManager.programs["line"], "pMatrix"), 1, false, &camera.pMatrix[0][0]);
		if (renderManager.renderingMode == RenderManager::RENDERING_MODE_LINE) {
			glUniform1i(glGetUniformLocation(renderManager.programs["line"], "useHatching"), 0);
		}
		else {
			glUniform1i(glGetUniformLocation(renderManager.programs["line"], "useHatching"), 1);
		}

		glUniform1i(glGetUniformLocation(renderManager.programs["line"], "tex0"), 1);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, renderManager.fragDataTex[0]);

		glUniform1i(glGetUniformLocation(renderManager.programs["line"], "tex1"), 2);
		glActiveTexture(GL_TEXTURE2);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, renderManager.fragDataTex[1]);

		glUniform1i(glGetUniformLocation(renderManager.programs["line"], "tex2"), 3);
		glActiveTexture(GL_TEXTURE3);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, renderManager.fragDataTex[2]);

		glUniform1i(glGetUniformLocation(renderManager.programs["line"], "tex3"), 4);
		glActiveTexture(GL_TEXTURE4);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, renderManager.fragDataTex[3]);

		glUniform1i(glGetUniformLocation(renderManager.programs["line"], "depthTex"), 8);
		glActiveTexture(GL_TEXTURE8);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, renderManager.fragDepthTex);

		glUniform1i(glGetUniformLocation(renderManager.programs["line"], "hatchingTexture"), 5);
		glActiveTexture(GL_TEXTURE5);
		glBindTexture(GL_TEXTURE_3D, renderManager.hatchingTextures);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

		glBindVertexArray(renderManager.secondPassVAO);

		glDrawArrays(GL_QUADS, 0, 4);
		glBindVertexArray(0);
		glDepthFunc(GL_LEQUAL);
	}


	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Blur

	if (renderManager.renderingMode != RenderManager::RENDERING_MODE_LINE && renderManager.renderingMode != RenderManager::RENDERING_MODE_HATCHING) {
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		qglClearColor(QColor(0xFF, 0xFF, 0xFF));
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glDisable(GL_DEPTH_TEST);
		glDepthFunc(GL_ALWAYS);

		glUseProgram(renderManager.programs["blur"]);
		glUniform2f(glGetUniformLocation(renderManager.programs["blur"], "pixelSize"), 2.0f / this->width(), 2.0f / this->height());
		//printf("pixelSize loc %d\n", glGetUniformLocation(vboRenderManager.programs["blur"], "pixelSize"));

		glUniform1i(glGetUniformLocation(renderManager.programs["blur"], "tex0"), 1);//COLOR
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, renderManager.fragDataTex[0]);

		glUniform1i(glGetUniformLocation(renderManager.programs["blur"], "tex1"), 2);//NORMAL
		glActiveTexture(GL_TEXTURE2);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, renderManager.fragDataTex[1]);

		/*glUniform1i(glGetUniformLocation(renderManager.programs["blur"], "tex2"), 3);
		glActiveTexture(GL_TEXTURE3);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, renderManager.fragDataTex[2]);*/

		glUniform1i(glGetUniformLocation(renderManager.programs["blur"], "depthTex"), 8);
		glActiveTexture(GL_TEXTURE8);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, renderManager.fragDepthTex);

		glUniform1i(glGetUniformLocation(renderManager.programs["blur"], "tex3"), 4);//AO
		glActiveTexture(GL_TEXTURE4);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, renderManager.fragAOTex);

		if (renderManager.renderingMode == RenderManager::RENDERING_MODE_SSAO) {
			glUniform1i(glGetUniformLocation(renderManager.programs["blur"], "ssao_used"), 1); // ssao used
		}
		else {
			glUniform1i(glGetUniformLocation(renderManager.programs["blur"], "ssao_used"), 0); // no ssao
		}

		glBindVertexArray(renderManager.secondPassVAO);

		glDrawArrays(GL_QUADS, 0, 4);
		glBindVertexArray(0);
		glDepthFunc(GL_LEQUAL);

	}

	// REMOVE
	glActiveTexture(GL_TEXTURE0);
}

void GLWidget3D::clearSketch() {
	sketch = QImage(this->width(), this->height(), QImage::Format_ARGB32);

	//sketch = QImage(this->width(), this->height(), QImage::Format_RGB888);
	//sketch.fill(qRgba(255, 255, 255, 255));

	strokes.clear();
	current_stroke.clear();
	sketchGraph.clear();
}

void GLWidget3D::undo() {
	if (strokes.size() > 0) {
		strokes.erase(strokes.begin() + strokes.size() - 1);
		current_stroke.clear();
	}

	sketch = QImage(this->width(), this->height(), QImage::Format_ARGB32);

	QPainter painter(&sketch);
	painter.setPen(QPen(QBrush(QColor(0, 0, 0)), 1));
	painter.setRenderHint(QPainter::Antialiasing);
	painter.setRenderHint(QPainter::HighQualityAntialiasing);

	for (int i = 0; i < strokes.size(); ++i) {
		if (strokes[i].size() <= 1) continue;

		QPoint pt1(strokes[i][0].x, height() - strokes[i][0].y);
		for (int k = 1; k < strokes[i].size(); ++k) {
			QPoint pt2(strokes[i][k].x, height() - strokes[i][k].y);
			painter.drawLine(pt1, pt2);
			pt1 = pt2;
		}
	}
}

void GLWidget3D::loadSketch(const QString& filename) {
	QFile file(filename);
	if (!file.open(QIODevice::ReadOnly)) return;

	clearSketch();

	QPainter painter(&sketch);
	painter.setPen(QPen(QBrush(QColor(0, 0, 0)), 1));
	painter.setRenderHint(QPainter::Antialiasing);
	painter.setRenderHint(QPainter::HighQualityAntialiasing);
	//painter.setPen(QPen(QBrush(QColor(0, 0, 0)), 3, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));

	QTextStream in(&file);
	while (!in.atEnd()) {
		QString str;
		in >> str;
		QStringList list = str.split("|");

		std::vector<glm::vec2> stroke;
		glm::vec2 pt1;

		for (int k = 0; k < list.size(); ++k) {
			QStringList xydata = list[k].split(",");
			if (xydata.size() == 2) {
				if (stroke.size() == 0) {
					pt1 = glm::vec2(xydata[0].toFloat(), xydata[1].toFloat());
					stroke.push_back(pt1);
				}
				else {
					glm::vec2 pt2 = glm::vec2(xydata[0].toFloat(), xydata[1].toFloat());
					stroke.push_back(pt2);
					painter.drawLine(QPoint(pt1.x, height() - pt1.y), QPoint(pt2.x, height() - pt2.y));
					pt1 = pt2;
				}
			}
		}

		strokes.push_back(stroke);
	}
}

void GLWidget3D::saveSketch(const QString& filename) {
	QFile file(filename);
	if (!file.open(QIODevice::WriteOnly)) return;

	QTextStream out(&file);
	for (int i = 0; i < strokes.size(); ++i) {
		for (int j = 0; j < strokes[i].size(); ++j) {
			if (j > 0) out << "|";
			out << strokes[i][j].x << "," << strokes[i][j].y;
		}
		out << "\n";
	}
	file.close();
}

/**
* Draw the scene.
*/
void GLWidget3D::drawScene() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glDepthMask(true);

	renderManager.renderAll();
}

void GLWidget3D::drawLine(const QPoint& startPoint, const QPoint& endPoint) {
	QPoint pt1(startPoint.x(), startPoint.y());
	QPoint pt2(endPoint.x(), endPoint.y());

	QPainter painter(&sketch);
	painter.setPen(QPen(QBrush(QColor(0, 0, 0)), 1));
	painter.setRenderHint(QPainter::Antialiasing);
	painter.setRenderHint(QPainter::HighQualityAntialiasing);
	//painter.setPen(QPen(QBrush(QColor(0, 0, 0)), 3, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));

	current_stroke.push_back(glm::vec2(((float)endPoint.x() / width() * 2 - 1) * camera.aspect(), 1.0f - (float)endPoint.y() / height() * 2));
	strokes.back().push_back(glm::vec2(((float)endPoint.x() / width() * 2 - 1) * camera.aspect(), 1.0f - (float)endPoint.y() / height() * 2));

	painter.drawLine(pt1, pt2);
}

void GLWidget3D::computeVanishingPoints(std::vector<sketch::VanishingPoint>& pv) {
	pv.resize(3);

	glm::vec4 p1(-50, 0, -50, 1);
	glm::vec4 p2(-50, 0, 50, 1);
	glm::vec4 p3(50, 0, 50, 1);
	glm::vec4 p4(50, 0, -50, 1);
	glm::vec4 p5(-50, 100, -50, 1);
	glm::vec4 p6(-50, 100, 50, 1);
	glm::vec4 p7(50, 100, 50, 1);
	glm::vec4 p8(50, 100, -50, 1);

	p1 = camera.mvpMatrix * p1;
	p2 = camera.mvpMatrix * p2;
	p3 = camera.mvpMatrix * p3;
	p4 = camera.mvpMatrix * p4;
	p5 = camera.mvpMatrix * p5;
	p6 = camera.mvpMatrix * p6;
	p7 = camera.mvpMatrix * p7;
	p8 = camera.mvpMatrix * p8;

	glm::vec2 pp1(p1.x / p1.w * camera.aspect(), p1.y / p1.w);
	glm::vec2 pp2(p2.x / p2.w * camera.aspect(), p2.y / p2.w);
	glm::vec2 pp3(p3.x / p3.w * camera.aspect(), p3.y / p3.w);
	glm::vec2 pp4(p4.x / p4.w * camera.aspect(), p4.y / p4.w);
	glm::vec2 pp5(p5.x / p5.w * camera.aspect(), p5.y / p5.w);
	glm::vec2 pp6(p6.x / p6.w * camera.aspect(), p6.y / p6.w);
	glm::vec2 pp7(p7.x / p7.w * camera.aspect(), p7.y / p7.w);
	glm::vec2 pp8(p8.x / p8.w * camera.aspect(), p8.y / p8.w);

	float tab, tcd;
	if (utils::segmentSegmentIntersect(pp2, pp3, pp5, pp8, &tab, &tcd, false, pv[0].pt)) {
		pv[0].type = sketch::VanishingPoint::TYPE_FINITE;
	}
	else {
		pv[0].type = sketch::VanishingPoint::TYPE_INFINITE;
		pv[0].pt = pp2 - pp3;
		pv[0].pt /= glm::length(pv[0].pt);
	}
	if (utils::segmentSegmentIntersect(pp4, pp3, pp5, pp6, &tab, &tcd, false, pv[1].pt)) {
		pv[1].type = sketch::VanishingPoint::TYPE_FINITE;
	}
	else {
		pv[1].type = sketch::VanishingPoint::TYPE_INFINITE;
		pv[1].pt = pp4 - pp3;
		pv[1].pt /= glm::length(pv[1].pt);
	}
	if (utils::segmentSegmentIntersect(pp6, pp2, pp8, pp4, &tab, &tcd, false, pv[2].pt)) {
		pv[2].type = sketch::VanishingPoint::TYPE_FINITE;
	}
	else {
		pv[2].type = sketch::VanishingPoint::TYPE_FINITE;
		pv[2].pt = pp7 - pp3;
		pv[2].pt /= glm::length(pv[2].pt);
	}
}

void GLWidget3D::reconstruct() {
	sketchGraph.reconstruct(&camera, width(), height());

	std::vector<Vertex> vertices;
	for (int i = 0; i < sketchGraph.faces3d.size(); ++i) {
		glutils::drawPolygon(sketchGraph.faces3d[i].points, glm::vec4(1, 0, 0, 1), vertices);
	}
	renderManager.addObject("object", "", vertices, true);
}

void GLWidget3D::resizeSketch(int width, int height) {
	QImage newImage(width, height, QImage::Format_ARGB32);
	newImage.fill(qRgba(255, 255, 255, 0));
	QPainter painter(&newImage);

	painter.drawImage((width - sketch.width()) * 0.5, (height - sketch.height()) * 0.5, sketch);
	sketch = newImage;
}

void GLWidget3D::keyPressEvent(QKeyEvent *e) {
	ctrlPressed = false;
	shiftPressed = false;
	altPressed = false;

	switch (e->key()) {
	case Qt::Key_Control:
		ctrlPressed = true;
		break;
	case Qt::Key_Shift:
		shiftPressed = true;
		break;
	case Qt::Key_Alt:
		altPressed = true;
		break;
	default:
		break;
	}
}

void GLWidget3D::keyReleaseEvent(QKeyEvent* e) {
	ctrlPressed = false;
	shiftPressed = false;
	altPressed = false;
}

/**
 * This event handler is called when the mouse press events occur.
 */
void GLWidget3D::mousePressEvent(QMouseEvent* e) {
	lastPos = e->pos();
	camera.mousePress(e->x(), e->y());

	if (e->buttons() & Qt::LeftButton) {
		std::vector<glm::vec2> pts;
		pts.push_back(glm::vec2(((float)e->x() / width() * 2 - 1) * camera.aspect(), 1.0f - (float)e->y() / height() * 2));
		current_stroke.push_back(glm::vec2(((float)e->x() / width() * 2 - 1) * camera.aspect(), 1.0f - (float)e->y() / height() * 2));
		strokes.push_back(pts);
	}
}

/**
 * This event handler is called when the mouse release events occur.
 */
void GLWidget3D::mouseReleaseEvent(QMouseEvent* e) {
	if (e->button() == Qt::LeftButton) {
		computeVanishingPoints(sketchGraph.pv);

		sketchGraph.addStroke(strokes.back());
		current_stroke.clear();
	}

	update();
}

/**
 * This event handler is called when the mouse move events occur.
 */
void GLWidget3D::mouseMoveEvent(QMouseEvent* e) {
	if (e->buttons() & Qt::RightButton) {
		if (shiftPressed) { // Move
			camera.move(e->x(), e->y());
		}
		else { // Rotate
			camera.rotate(e->x(), e->y());
		}
	}
	else if (e->buttons() & Qt::LeftButton) {
		drawLine(lastPos, e->pos());
	}

	lastPos = e->pos();
	update();
}

void GLWidget3D::wheelEvent(QWheelEvent* e) {
	// zoom
	camera.zoom(e->delta()); 
	update();
}

/**
 * This function is called once before the first call to paintGL() or resizeGL().
 */
void GLWidget3D::initializeGL() {
	// init glew
	GLenum err = glewInit();
	if (err != GLEW_OK) {
		std::cout << "Error: " << glewGetErrorString(err) << std::endl;
	}

	if (glewIsSupported("GL_VERSION_4_2"))
		printf("Ready for OpenGL 4.2\n");
	else {
		printf("OpenGL 4.2 not supported\n");
		exit(1);
	}
	const GLubyte* text = glGetString(GL_VERSION);
	printf("VERSION: %s\n", text);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	glEnable(GL_TEXTURE_2D);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);

	glTexGenf(GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
	glTexGenf(GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
	glDisable(GL_TEXTURE_2D);

	glEnable(GL_TEXTURE_3D);
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glDisable(GL_TEXTURE_3D);

	glEnable(GL_TEXTURE_2D_ARRAY);
	glTexParameterf(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameterf(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glDisable(GL_TEXTURE_2D_ARRAY);

	////////////////////////////////
	renderManager.init(true, 8192);
	renderManager.resize(this->width(), this->height());
	renderManager.renderingMode = RenderManager::RENDERING_MODE_BASIC;

	glUniform1i(glGetUniformLocation(renderManager.programs["ssao"], "tex0"), 0);//tex0: 0

	camera.xrot = 10.0f;
	camera.yrot = -45.0f;
	camera.zrot = 0.0f;
	camera.pos = glm::vec3(0, 40, 158);

	//clearImage();

	std::vector<Vertex> vertices;
	glutils::drawGrid(200, 200, 10, glm::vec4(0, 0, 1, 1), glm::vec4(0.9, 0.9, 1.0, 1), glm::rotate(glm::mat4(), -(float)M_PI * 0.5f, glm::vec3(1, 0, 0)), vertices);
	renderManager.addObject("axis", "", vertices, false);

	vertices.clear();
	glutils::drawQuad(50, 50, glm::vec4(1, 0, 1, 1), glm::translate(glm::mat4(), glm::vec3(-25, 25, 0)), vertices);
	renderManager.addObject("box", "", vertices, false);
	
	// sketch imageを初期化
	sketch = QImage(this->width(), this->height(), QImage::Format_ARGB32);
}

/**
 * This function is called whenever the widget has been resized.
 */
void GLWidget3D::resizeGL(int width, int height) {
	height = height ? height : 1;

	glViewport(0, 0, (GLint)width, (GLint)height);
	camera.updatePMatrix(width, height);
	renderManager.resize(width, height);
	renderManager.updateShadowMap(this, light_dir, light_mvpMatrix);

	// sketch imageを更新
	resizeSketch(width, height);
}

/**
 * This function is called whenever the widget needs to be painted.
 */
void GLWidget3D::paintEvent(QPaintEvent* e) {
	// OpenGLで描画
	makeCurrent();

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();

	render();

	// OpenGLの設定を元に戻す
	glShadeModel(GL_FLAT);
	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	// draw sketch
	QPainter painter(this);
	painter.setOpacity(0.5);
	//painter.drawImage(0, 0, sketch);

	// draw a current stroke
	painter.setOpacity(1.0f);
	painter.setRenderHint(QPainter::Antialiasing);
	painter.setRenderHint(QPainter::HighQualityAntialiasing);
	painter.setPen(QPen(QBrush(QColor(0, 0, 0)), 1));
	for (int i = 1; i < current_stroke.size(); ++i) {
		painter.drawLine(current_stroke[i - 1].x * 0.5f * height() + width() * 0.5f, (1.0f - current_stroke[i - 1].y) * 0.5f * height(), current_stroke[i].x * 0.5f * height() + width() * 0.5f, (1.0f - current_stroke[i].y) * 0.5f * height());
	}

	// draw face
	painter.setOpacity(1.0f);
	for (int i = 0; i < sketchGraph.faces.size(); ++i) {
		painter.setPen(QPen(QBrush(QColor(0, 0, 0)), 1));

		if (sketchGraph.faces[i].pv[0] == 0 && sketchGraph.faces[i].pv[1] == 1) {
			painter.setBrush(QBrush(QColor(255, 255, sketchGraph.faces[i].validness * 255)));
		}
		else if (sketchGraph.faces[i].pv[0] == 0 && sketchGraph.faces[i].pv[1] == 2) {
			painter.setBrush(QBrush(QColor(255, sketchGraph.faces[i].validness * 255, 255)));
		}
		else if (sketchGraph.faces[i].pv[0] == 1 && sketchGraph.faces[i].pv[1] == 2) {
			painter.setBrush(QBrush(QColor(sketchGraph.faces[i].validness * 255, 255, 255)));
		}
		else {
			painter.setBrush(QBrush(QColor(0, 0, 0)));
		}
		QPolygon polygon;
		for (int k = 0; k < sketchGraph.faces[i].points.size(); ++k) {
			polygon.push_back(QPoint(sketchGraph.faces[i].points[k].x, height() - sketchGraph.faces[i].points[k].y));
		}
		painter.drawPolygon(polygon);
	}

	// draw edge
	painter.setOpacity(1.0f);
	sketch::EdgeIter ei, eend;
	for (boost::tie(ei, eend) = boost::edges(sketchGraph.graph); ei != eend; ++ei) {
		QPen pen;
		if (sketchGraph.graph[*ei]->g_values[0] > sketchGraph.graph[*ei]->g_values[1] && sketchGraph.graph[*ei]->g_values[0] > sketchGraph.graph[*ei]->g_values[2]) {
			pen = QPen(QBrush(QColor(255, 0, 0)), 3);
		}
		else if (sketchGraph.graph[*ei]->g_values[1] > sketchGraph.graph[*ei]->g_values[2]) {
			pen = QPen(QBrush(QColor(0, 255, 0)), 3);
		}
		else if (sketchGraph.graph[*ei]->g_values[2] > 0) {
			pen = QPen(QBrush(QColor(0, 0, 255)), 3);
		}
		else {
			pen = QPen(QBrush(QColor(0, 0, 0)), 3);
		}

		//QPen pen(QBrush(QColor(sketchGraph.graph[*ei]->g_values[0] * 255, sketchGraph.graph[*ei]->g_values[1] * 255, sketchGraph.graph[*ei]->g_values[2] * 255)), 3);
		painter.setPen(pen);
		
		for (int k = 1; k < sketchGraph.graph[*ei]->points.size(); ++k) {
			painter.drawLine(sketchGraph.graph[*ei]->points[k - 1].x * 0.5f * height() + width() * 0.5f, (1.0f - sketchGraph.graph[*ei]->points[k - 1].y) * 0.5f * height() , sketchGraph.graph[*ei]->points[k].x * 0.5f * height() + width() * 0.5f, (1.0f - sketchGraph.graph[*ei]->points[k].y) * 0.5f * height());
		}
	}

	// draw junction type
	painter.setOpacity(1.0f);
	sketch::VertexIter vi, vend;
	painter.setFont(QFont("Bavaria", 12, 4));
	painter.setPen(QPen(QBrush(QColor(0, 0, 0)), 1));
	for (boost::tie(vi, vend) = boost::vertices(sketchGraph.graph); vi != vend; ++vi) {
		sketch::VertexPtr v = sketchGraph.graph[*vi];
		QString letter;
		if (v->type == sketch::SketchVertex::TYPE_ISOLATED) {
			letter = "I";
		}
		else if (v->type == sketch::SketchVertex::TYPE_L) {
			letter = "L";
		}
		else if (v->type == sketch::SketchVertex::TYPE_T) {
			letter = "T";
		}
		else if (v->type == sketch::SketchVertex::TYPE_Y) {
			letter = "Y";
		}
		else if (v->type == sketch::SketchVertex::TYPE_E) {
			letter = "E";
		}
		else if (v->type == sketch::SketchVertex::TYPE_X) {
			letter = "X";
		}
		painter.drawText(v->pt.x * 0.5f * height() + width() * 0.5f - 5, (1.0f - v->pt.y) * 0.5f * height() - 5, 30, 30, Qt::AlignHCenter | Qt::AlignVCenter, letter);
	}

	painter.end();

	glEnable(GL_DEPTH_TEST);
}
