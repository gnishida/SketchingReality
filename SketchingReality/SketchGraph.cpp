#include "SketchGraph.h"
#include "Utils.h"
#include <opencv2/opencv.hpp>
#include "Camera.h"

#ifndef M_PI
#define M_PI 3.1415926535
#endif

namespace sketch {
	const float STRAIGHT_THRESHOLD = 0.866f; // 30度に相当 (30度以上ずれた線分は、カーブとみなす)
	const float DIST_TO_JUNCTION_THRESHOLD = 8.0f;

	SketchGraph* graphPtr;
	std::vector<Face>* facesPtr;

	struct output_visitor : public boost::planar_face_traversal_visitor {
		void begin_face() {
			facesPtr->push_back(Face());
		}

		void end_face() {
			// 時計回りの面は除外する
			if (utils::area(facesPtr->back().points) < 0) {
				facesPtr->erase(facesPtr->begin() + facesPtr->size() - 1);
			}
		}
	};

	//Vertex visitor
	struct vertex_output_visitor : public output_visitor {
		template <typename Vertex>
		void next_vertex(Vertex v) {
		}

		template <typename Edge>
		void next_edge(Edge e) {
			facesPtr->back().edges.push_back(e);

			if (facesPtr->back().points.size() == 0) {
				facesPtr->back().points.insert(facesPtr->back().points.end(), graphPtr->graph[e]->points.begin(), graphPtr->graph[e]->points.end());
			}
			else {
				if (facesPtr->back().edges.size() == 2) {
					float dist1 = glm::length(graphPtr->graph[e]->points[0] - facesPtr->back().points.back());
					float dist2 = glm::length(graphPtr->graph[e]->points.back() - facesPtr->back().points.back());
					float dist3 = glm::length(graphPtr->graph[e]->points[0] - facesPtr->back().points[0]);
					float dist4 = glm::length(graphPtr->graph[e]->points.back() - facesPtr->back().points[0]);

					if (std::min(dist1, dist2) > std::min(dist3, dist4)) {
						std::reverse(facesPtr->back().points.begin(), facesPtr->back().points.end());
					}
				}

				if (glm::length(graphPtr->graph[e]->points[0] - facesPtr->back().points.back()) < glm::length(graphPtr->graph[e]->points.back() - facesPtr->back().points.back())) {
					facesPtr->back().points.insert(facesPtr->back().points.end(), graphPtr->graph[e]->points.begin(), graphPtr->graph[e]->points.end());
				}
				else {
					std::reverse(graphPtr->graph[e]->points.begin(), graphPtr->graph[e]->points.end());
					facesPtr->back().points.insert(facesPtr->back().points.end(), graphPtr->graph[e]->points.begin(), graphPtr->graph[e]->points.end());
				}
			}
		}
	};

	VanishingPoint::VanishingPoint(int type, const glm::vec2& pt) {
		this->type = type;
		this->pt = pt;
	}

	SketchGraph::SketchGraph() {
	}

	void SketchGraph::clear() {
		graph.clear();
		faces.clear();
	}

	void SketchGraph::addStroke(const std::vector<glm::vec2>& stroke) {
		if (stroke.size() <= 1) return;

		//std::cout << "------------------------------" << std::endl;
		//std::cout << "Stroke: " << glm::to_string(stroke[0]) << " - " << glm::to_string(stroke.back()) << std::endl;

		// determine the type of the edge
		int edge_type;
		if (isStraight(stroke)) {
			edge_type = SketchEdge::TYPE_STRAIGHT;
		}
		else {
			edge_type = SketchEdge::TYPE_CURVED;

			// not implemented yet
		}

		// find junctions with existing edges
		VertexDesc v_desc1, v_desc2;
		findJunction(stroke[0], v_desc1, DIST_TO_JUNCTION_THRESHOLD);
		//std::cout << "v1: " << v_desc1 << ", " << glm::to_string(graph[v_desc1]->pt) << std::endl;
		findJunction(stroke.back(), v_desc2, DIST_TO_JUNCTION_THRESHOLD);
		//std::cout << "v2: " << v_desc2 << ", " << glm::to_string(graph[v_desc2]->pt) << std::endl;

		// add a new edge
		EdgePtr e = EdgePtr(new SketchEdge(edge_type, stroke));
		std::pair<EdgeDesc, bool> edge_pair = boost::add_edge(v_desc1, v_desc2, graph);
		graph[edge_pair.first] = e;

		// group the edge based on vanishing points
		groupEdge(edge_pair.first);

		// update the junction type
		updateJunctionType(v_desc1);
		updateJunctionType(v_desc2);

		// extrace faces
		extractFaces(faces);

		// compute face validness
		for (int i = 0; i < faces.size(); ++i) {
			float v1 = computeFaceValidness(faces[i], pv[0], pv[1]);
			float v2 = computeFaceValidness(faces[i], pv[0], pv[2]);
			float v3 = computeFaceValidness(faces[i], pv[1], pv[2]);

			faces[i].pv.resize(2);
			if (v1 > v2 && v1 > v3) {
				faces[i].validness = v1;
				faces[i].pv[0] = 0;
				faces[i].pv[1] = 1;
			}
			else if (v2 > v3) {
				faces[i].validness = v2;
				faces[i].pv[0] = 0;
				faces[i].pv[1] = 2;
			}
			else if (v3 > 0) {
				faces[i].validness = v3;
				faces[i].pv[0] = 1;
				faces[i].pv[1] = 2;
			}
			else {
				faces[i].validness = 0.0f;
				faces[i].pv[0] = 0;
				faces[i].pv[1] = 0;
			}
		}
	}

	bool SketchGraph::isStraight(const std::vector<glm::vec2>& stroke) {
		if (stroke.size() <= 2) return true;

		glm::vec2 v1 = stroke.back() - stroke[0];
		v1 /= glm::length(v1);
		for (int i = 1; i < stroke.size() - 1; ++i) {
			glm::vec2 v2 = stroke[i] - stroke[0];
			v2 /= glm::length(v2);
			if (glm::dot(v1, v2) < STRAIGHT_THRESHOLD) return false;
		}

		return true;
	}
	
	/**
	 * 指定された点を既存グラフに接続するための交点を探す。
	 * もし既存エッジに接続する場合は、既存エッジを分割する。
	 * 近くに頂点・エッジがない場合は、新規頂点を追加する。
	 */
	void SketchGraph::findJunction(const glm::vec2& pt, VertexDesc& v_desc, float threshold) {
		if (findVertex(pt, v_desc, DIST_TO_JUNCTION_THRESHOLD)) {
			// do nothing
			//std::cout << "snap to vertex" << std::endl;
		}
		else if (findEdge(pt, v_desc, DIST_TO_JUNCTION_THRESHOLD)) {
			// do nothing
			//std::cout << "snap to edge" << std::endl;
		}
		else {
			// add a new vertex
			v_desc = boost::add_vertex(graph);
			graph[v_desc] = VertexPtr(new SketchVertex(pt));
			graph[v_desc]->type = SketchVertex::TYPE_ISOLATED;
			//std::cout << "new veretx: " << v_desc << ", " << glm::to_string(pt) << std::endl;
		}
	}

	bool SketchGraph::findVertex(const glm::vec2& pt, VertexDesc& v, float threshold) {
		float min_dist = std::numeric_limits<float>::max();
		bool found = false;

		VertexIter vi, vend;
		for (boost::tie(vi, vend) = boost::vertices(graph); vi != vend; ++vi) {
			float dist = glm::length(graph[*vi]->pt - pt);
			if (dist < threshold && dist < min_dist) {
				v = *vi;
				min_dist = dist;
				found = true;
			}
		}

		return found;
	}

	bool SketchGraph::findEdge(const glm::vec2& pt, VertexDesc& v, float threshold) {
		//std::cout << "finding edge: " << glm::to_string(pt) << std::endl;

		float min_dist = std::numeric_limits<float>::max();
		EdgeDesc min_e;
		bool found = false;
		int index1, index2;
		glm::vec2 junctionPt;

		EdgeIter ei, eend;
		for (boost::tie(ei, eend) = boost::edges(graph); ei != eend; ++ei) {
			for (int i = 1; i < graph[*ei]->points.size(); i++) {
				glm::vec2 intPt;
				float dist = utils::pointSegmentDistance(graph[*ei]->points[i - 1], graph[*ei]->points[i], pt, intPt);
				if (dist < threshold && dist < min_dist) {
					min_dist = dist;
					min_e = *ei;
					junctionPt = intPt;
					index1 = i;
					index2 = i + 1;
					found = true;

					//std::cout << "dist=" << dist << ", index1=" << index1 << ", index2=" << index2 << ", junctionPt=" << glm::to_string(junctionPt) << std::endl;
				}
			}
		}

		if (found) {
			//std::cout << "found: " << glm::to_string(junctionPt) << std::endl;


			// obtain src and tgt
			VertexDesc src_desc = boost::source(min_e, graph);
			VertexDesc tgt_desc = boost::target(min_e, graph);

			// add a new vertex
			v = boost::add_vertex(graph);
			graph[v] = VertexPtr(new SketchVertex(junctionPt));
			graph[v]->type = SketchVertex::TYPE_LINE;

			// split the points into two segments
			std::vector<glm::vec2> stroke1, stroke2;
			for (int i = 0; i < index1; ++i) {
				stroke1.push_back(graph[min_e]->points[i]);
			}
			stroke1.push_back(junctionPt);
			stroke2.push_back(junctionPt);
			for (int i = index2; i < graph[min_e]->points.size(); ++i) {
				stroke2.push_back(graph[min_e]->points[i]);
			}

			// create two segment edges
			EdgePtr e1 = EdgePtr(new SketchEdge(graph[min_e]->type, stroke1));
			EdgePtr e2 = EdgePtr(new SketchEdge(graph[min_e]->type, stroke2));
			e1->g_values = graph[min_e]->g_values;
			e2->g_values = graph[min_e]->g_values;
			if (glm::length(graph[min_e]->points[0] - graph[src_desc]->pt) < glm::length(graph[min_e]->points[0] - graph[tgt_desc]->pt)) {
				std::pair<EdgeDesc, bool> edge_pair1 = boost::add_edge(src_desc, v, graph);
				graph[edge_pair1.first] = e1;
				std::pair<EdgeDesc, bool> edge_pair2 = boost::add_edge(v, tgt_desc, graph);
				graph[edge_pair2.first] = e2;
			}
			else {
				std::pair<EdgeDesc, bool> edge_pair1 = boost::add_edge(tgt_desc, v, graph);
				graph[edge_pair1.first] = e1;
				std::pair<EdgeDesc, bool> edge_pair2 = boost::add_edge(v, src_desc, graph);
				graph[edge_pair2.first] = e2;
			}
			
			boost::remove_edge(min_e, graph);
		}
		
		return found;
	}

	void SketchGraph::updateJunctionType(VertexDesc v_desc) {
		int degree = boost::out_degree(v_desc, graph);
		if (degree <= 1) {
			graph[v_desc]->type = SketchVertex::TYPE_ISOLATED;
		}
		else if (degree == 2) {
			graph[v_desc]->type = SketchVertex::TYPE_L;
		}
		else if (degree >= 4) {
			graph[v_desc]->type = SketchVertex::TYPE_X;
		}
		else {
			if (graph[v_desc]->type == SketchVertex::TYPE_LINE) {
				graph[v_desc]->type = SketchVertex::TYPE_T;
			}
			else {
				std::vector<glm::vec2> out_vecs;
				float total_theta = 0.0f;

				OutEdgeIter ei, ei_end;
				for (boost::tie(ei, ei_end) = boost::out_edges(v_desc, graph); ei != ei_end; ++ei) {
					VertexDesc tgt_desc = boost::target(*ei, graph);
					glm::vec2 vec = graph[tgt_desc]->pt - graph[v_desc]->pt;
					vec /= glm::length(vec);
					out_vecs.push_back(vec);
				}

				total_theta += fabs(acosf(glm::dot(out_vecs[0], out_vecs[1])));
				total_theta += fabs(acosf(glm::dot(out_vecs[0], out_vecs[2])));
				total_theta += fabs(acosf(glm::dot(out_vecs[1], out_vecs[2])));

				if (total_theta >= M_PI * 2.0f - 1.0f) {
					graph[v_desc]->type = SketchVertex::TYPE_Y;
				}
				else {
					graph[v_desc]->type = SketchVertex::TYPE_E;
				}
			}
		}

	}

	void SketchGraph::groupEdge(EdgeDesc e_desc) {
		graph[e_desc]->g_values.resize(3);
		graph[e_desc]->g_values[0] = g(e_desc, pv[0]);
		graph[e_desc]->g_values[1] = g(e_desc, pv[1]);
		graph[e_desc]->g_values[2] = g(e_desc, pv[2]);
	}

	/**
	 * This equation is defined in Equation (3) in the Sketching Reality paper.
	 */
	float SketchGraph::g(EdgeDesc e_desc, const VanishingPoint& vp) {
		float dist = d(e_desc, vp);
		if (dist < 0.35f) {
			return 1 - dist / 0.35;
		}
		else {
			return 0;
		}
	}

	float SketchGraph::d(EdgeDesc e_desc, const VanishingPoint& vp) {
		VertexDesc src = boost::source(e_desc, graph);
		VertexDesc tgt = boost::target(e_desc, graph);

		if (vp.type == VanishingPoint::TYPE_INFINITE) {
			glm::vec2 vec = graph[src]->pt - graph[tgt]->pt;

			float rad = acosf(glm::dot(vp.pt, vec) / glm::length(vp.pt) / glm::length(vec));
			if (rad > M_PI * 0.5) rad = M_PI - rad;
			return rad;
		}
		else {
			glm::vec2 midPt = (graph[src]->pt + graph[tgt]->pt) * 0.5f;

			float rad = acosf(glm::dot(graph[src]->pt - midPt, vp.pt - midPt) / glm::length(graph[src]->pt - midPt) / glm::length(vp.pt - midPt));
			if (rad > M_PI * 0.5) rad = M_PI - rad;
			return rad;
		}
	}

	/**
	 * Compute the validness of the face in terms of two selected vanishing points.
	 * See the definition of P(F valid|e_0,...,e_k,p_va,p_vb) in p.8 in the Sketching Reality paper.
	 */
	float SketchGraph::computeFaceValidness(const Face& face, const VanishingPoint& vp1, const VanishingPoint& vp2) {
		float max_P_vp1 = 0.0f;
		float max_P_vp2 = 0.0f;
		float P = 1.0f;

		for (int k = 0; k < face.edges.size(); ++k) {
			max_P_vp1 = std::max(max_P_vp1, g(face.edges[k], vp1));
			max_P_vp2 = std::max(max_P_vp2, g(face.edges[k], vp2));
			P *= std::max(g(face.edges[k], vp1), g(face.edges[k], vp2));
		}

		P = powf(P, 1.0f / face.edges.size());

		return max_P_vp1 * max_P_vp2 * P;
	}

	void SketchGraph::extractFaces(std::vector<Face>& faces) {
		graphPtr = this;
		faces.clear();
		facesPtr = &faces;

		typedef std::vector<EdgeDesc> tEdgeDescriptorVector;
		std::vector<tEdgeDescriptorVector> embedding(boost::num_vertices(graph));
		buildEmbedding(embedding);

		//Create edge index property map?	
		typedef std::map<EdgeDesc, size_t> EdgeIndexMap;
		EdgeIndexMap mapEdgeIdx;
		boost::associative_property_map<EdgeIndexMap> pmEdgeIndex(mapEdgeIdx);
		EdgeIter ei, ei_end;
		int edge_count = 0;
		for (boost::tie(ei, ei_end) = boost::edges(graph); ei != ei_end; ++ei) {
			mapEdgeIdx.insert(std::make_pair(*ei, edge_count++));
		}

		vertex_output_visitor v_vis;
		boost::planar_face_traversal(graph, &embedding[0], v_vis, pmEdgeIndex);
	}

	void SketchGraph::buildEmbedding(std::vector<std::vector<EdgeDesc> >& embedding) {
		embedding.clear();

		VertexIter vi, vend;
		for (boost::tie(vi, vend) = boost::vertices(graph); vi != vend; ++vi) {
			std::map<float, EdgeDesc> edges;

			OutEdgeIter ei, eend;
			for (boost::tie(ei, eend) = boost::out_edges(*vi, graph); ei != eend; ++ei) {
				VertexDesc src = boost::source(*ei, graph);
				VertexDesc tgt = boost::target(*ei, graph);
				glm::vec2 vec = graph[tgt]->pt - graph[src]->pt;
				
				edges[-atan2f(vec.y, vec.x)] = *ei;
			}

			std::vector<EdgeDesc> edge_descs;
			for (auto it = edges.begin(); it != edges.end(); ++it) {
				edge_descs.push_back(it->second);
			}

			embedding.push_back(edge_descs);
		}
	}

	void SketchGraph::reconstruct(Camera* camera, int screen_width, int screen_height) {
		cv::Mat_<float> A(20, 12, 0.0f);
		
		std::vector<glm::vec2> p;
		VertexIter vi, vend;
		for (boost::tie(vi, vend) = boost::vertices(graph); vi != vend; ++vi) {
			p.push_back(glm::vec2(graph[*vi]->pt.x / screen_width * 2 - 1, graph[*vi]->pt.y / screen_height * 2 - 1));
			std::cout << "(" << graph[*vi]->pt.x << ", " << graph[*vi]->pt.y << ")" << std::endl;
		}

		////////////////// debug ///////////////////////////////////////////////
		for (int i = 0; i < pv.size(); ++i) {
			pv[i].pt.x = pv[i].pt.x / screen_width * 2 - 1;
			pv[i].pt.y = pv[i].pt.y / screen_width * 2 - 1;
			std::cout << "pv: (" << pv[i].pt.x << ", " << pv[i].pt.y << ")" << std::endl;
		}
		////////////////// debug ///////////////////////////////////////////////

		// projection constraints
		for (int i = 0; i < p.size(); ++i) {
			A(i * 2 + 0, i * 3 + 0) = 1;
			A(i * 2 + 0, i * 3 + 2) = -p[i].x / camera->pMatrix[0][0];
			A(i * 2 + 1, i * 3 + 1) = 1;
			A(i * 2 + 1, i * 3 + 2) = -p[i].y / camera->pMatrix[1][1];
		}

		// vanishing point constraints
		cv::Mat_<float> K(3, 1);
		K(0, 0) = glm::length(p[1] - p[0]) / glm::length(pv[2].pt - p[1]) / camera->f() * pv[2].pt.x;
		K(1, 0) = glm::length(p[1] - p[0]) / glm::length(pv[2].pt - p[1]) / camera->f() * pv[2].pt.y;
		K(2, 0) = glm::length(p[1] - p[0]) / glm::length(pv[2].pt - p[1]) / camera->f() * -camera->f();
		A(8, 0) = 1;
		A(8, 2) = K(0, 0);
		A(8, 3) = -1;
		A(9, 1) = 1;
		A(9, 2) = K(1, 0);
		A(9, 4) = -1;
		A(10, 2) = 1 + K(2, 0);
		A(10, 5) = -1;

		K(0, 0) = glm::length(p[1] - p[2]) / glm::length(pv[0].pt - p[1]) / camera->f() * pv[0].pt.x;
		K(1, 0) = glm::length(p[1] - p[2]) / glm::length(pv[0].pt - p[1]) / camera->f() * pv[0].pt.y;
		K(2, 0) = glm::length(p[1] - p[2]) / glm::length(pv[0].pt - p[1]) / camera->f() * -camera->f();
		A(11, 3) = -1;
		A(11, 6) = 1;
		A(11, 8) = K(0, 0);
		A(12, 4) = -1;
		A(12, 7) = 1;
		A(12, 8) = K(1, 0);
		A(13, 5) = -1;
		A(13, 8) = 1 + K(2, 0);

		K(0, 0) = glm::length(p[0] - p[3]) / glm::length(pv[0].pt - p[0]) / camera->f() * pv[0].pt.x;
		K(1, 0) = glm::length(p[0] - p[3]) / glm::length(pv[0].pt - p[0]) / camera->f() * pv[0].pt.y;
		K(2, 0) = glm::length(p[0] - p[3]) / glm::length(pv[0].pt - p[0]) / camera->f() * -camera->f();
		A(14, 0) = -1;
		A(14, 9) = 1;
		A(14, 11) = K(0, 0);
		A(15, 1) = -1;
		A(15, 10) = 1;
		A(15, 11) = K(1, 0);
		A(16, 2) = -1;
		A(16, 11) = 1 + K(2, 0);

		K(0, 0) = glm::length(p[2] - p[3]) / glm::length(pv[2].pt - p[0]) / camera->f() * pv[2].pt.x;
		K(1, 0) = glm::length(p[2] - p[3]) / glm::length(pv[2].pt - p[0]) / camera->f() * pv[2].pt.y;
		K(2, 0) = glm::length(p[2] - p[3]) / glm::length(pv[2].pt - p[0]) / camera->f() * -camera->f();
		A(17, 6) = -1;
		A(17, 9) = 1;
		A(17, 11) = K(0, 0);
		A(18, 7) = -1;
		A(18, 10) = 1;
		A(18, 11) = K(1, 0);
		A(19, 8) = -1;
		A(19, 11) = 1 + K(2, 0);

		cv::SVD svd(A);
		cv::Mat x = svd.vt.row(svd.vt.rows - 1);

		std::cout << A * x.t() << std::endl;

		std::cout << "3D points: " << std::endl;
		for (int i = 0; i < 4; ++i) {
			std::cout << "(" << x.at<float>(0, i * 3) << ", " << x.at<float>(0, i * 3 + 1) << ", " << -x.at<float>(0, i * 3 + 2) << ")" << std::endl;
		}

		std::cout << "Projected points: " << std::endl;
		for (int i = 0; i < 4; ++i) {
			glm::vec4 p(x.at<float>(0, i * 3), x.at<float>(0, i * 3 + 1), -x.at<float>(0, i * 3 + 2), 1);
			p = camera->pMatrix * p;

			glm::vec2 pp(screen_width * (p.x / p.w * 0.5 + 0.5), screen_height * (p.y / p.w * 0.5 + 0.5));
			std::cout << "(" << pp.x << ", " << pp.y << ")" << std::endl;
		}

		// test
		glm::vec4 p0(-50, 50, 0, 1);
		glm::vec4 p1(-50, 0, 0, 1);
		glm::vec4 p2(0, 0, 0, 1);
		glm::vec4 p3(0, 50, 0, 1);
		p0 = camera->mvMatrix * p0;
		p1 = camera->mvMatrix * p1;
		p2 = camera->mvMatrix * p2;
		p3 = camera->mvMatrix * p3;

		x.at<float>(0, 0) = p0.x;
		x.at<float>(0, 1) = p0.y;
		x.at<float>(0, 2) = -p0.z;
		x.at<float>(0, 3) = p1.x;
		x.at<float>(0, 4) = p1.y;
		x.at<float>(0, 5) = -p1.z;
		x.at<float>(0, 6) = p2.x;
		x.at<float>(0, 7) = p2.y;
		x.at<float>(0, 8) = -p2.z;
		x.at<float>(0, 9) = p3.x;
		x.at<float>(0, 10) = p3.y;
		x.at<float>(0, 11) = -p3.z;

		std::cout << "-------------------------------------------" << std::endl;
		std::cout << "Ideal result:" << std::endl;
		std::cout << A * x.t() << std::endl;

		std::cout << "Projected points: " << std::endl;
		for (int i = 0; i < 4; ++i) {
			glm::vec4 p(x.at<float>(0, i * 3), x.at<float>(0, i * 3 + 1), -x.at<float>(0, i * 3 + 2), 1);
			p = camera->pMatrix * p;

			glm::vec2 pp(screen_width * (p.x / p.w * 0.5 + 0.5), screen_height * (p.y / p.w * 0.5 + 0.5));
			std::cout << "(" << pp.x << ", " << pp.y << ")" << std::endl;
		}



		std::cout << "-------------------------------------------" << std::endl;
		std::cout << "v0: (" << p0.x << ", " << p0.y << ", " << p0.z << ")" << std::endl;
		std::cout << "v1: (" << p1.x << ", " << p1.y << ", " << p1.z << ")" << std::endl;

		std::cout << "u0: (" << p[0].x << ", " << p[0].y << ")" << std::endl;

		glm::vec4 pp0 = camera->pMatrix * p0;
		glm::vec4 pp1 = camera->pMatrix * p1;
		std::cout << "p0: (" << pp0.x / pp0.w << ", " << pp0.y / pp0.w << ")" << std::endl;
		std::cout << "p1: (" << pp1.x / pp1.w << ", " << pp1.y / pp1.w << ")" << std::endl;
		std::cout << "pv: (" << pv[2].pt.x << ", " << pv[2].pt.y << ")" << std::endl;

		std::cout << std::endl;
		std::cout << A << std::endl;
	}
}