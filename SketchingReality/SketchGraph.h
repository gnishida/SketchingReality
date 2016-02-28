#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/planar_face_traversal.hpp>
#include <boost/graph/boyer_myrvold_planar_test.hpp>
#include "SketchEdge.h"
#include "SketchVertex.h"

class Camera;

namespace sketch {
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, VertexPtr, EdgePtr> BGLGraph;
	typedef boost::graph_traits<BGLGraph>::vertex_descriptor VertexDesc;
	typedef boost::graph_traits<BGLGraph>::edge_descriptor EdgeDesc;
	typedef boost::graph_traits<BGLGraph>::vertex_iterator VertexIter;
	typedef boost::graph_traits<BGLGraph>::edge_iterator EdgeIter;
	typedef boost::graph_traits<BGLGraph>::out_edge_iterator OutEdgeIter;
	typedef boost::graph_traits<BGLGraph>::in_edge_iterator InEdgeIter;

	class VanishingPoint {
	public:
		static enum { TYPE_INFINITE = 0, TYPE_FINITE };

	public:
		int type;
		glm::vec2 pt;

	public:
		VanishingPoint() {}
		VanishingPoint(int type, const glm::vec2& pt);
	};

	class Face {
	public:
		std::vector<VertexDesc> vertices;
		std::vector<EdgeDesc> edges;
		std::vector<glm::vec2> points; // contour points
		float validness;
		std::vector<int> pv;	// vanishing points

	public:
		Face() : validness(0) {}
	};

	class Face3D {
	public:
		std::vector<glm::vec3> points;

	public:
		Face3D() {}
	};

	class SketchGraph {
	public:
		BGLGraph graph;
		std::vector<VanishingPoint> pv;	// vanishing points
		std::vector<Face> faces;
		std::vector<Face3D> faces3d;

	public:
		SketchGraph();

		void clear();
		void addStroke(const std::vector<glm::vec2>& stroke);

		bool isStraight(const std::vector<glm::vec2>& stroke);
		void findJunction(const glm::vec2& pt, VertexDesc& v, float threshold);
		bool findVertex(const glm::vec2& pt, VertexDesc& v, float threshold);
		bool findEdge(const glm::vec2& pt, VertexDesc& v, float threshold);
		void updateJunctionType(VertexDesc v_desc);
		void groupEdge(EdgeDesc e_desc);
		float g(EdgeDesc e_desc, const VanishingPoint& vp);
		float d(EdgeDesc e_desc, const VanishingPoint& vp);
		float computeFaceValidness(const Face& face, const VanishingPoint& vp1, const VanishingPoint& vp2);
		void extractFaces(std::vector<Face>& faces);
		void buildEmbedding(std::vector<std::vector<EdgeDesc> >& embedding);
		void reconstruct(Camera* camera, int screen_width, int screen_height);
	};

	typedef boost::shared_ptr<SketchGraph> GraphPtr;
}