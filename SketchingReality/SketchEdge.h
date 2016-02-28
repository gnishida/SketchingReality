#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <boost/shared_ptr.hpp>

namespace sketch {
	class SketchEdge {
	public:
		static enum { TYPE_STRAIGHT = 0, TYPE_CURVED };

	public:
		int type;
		std::vector<float> g_values; // g(e, pv), which is defined in Equation (3) in the Sketching Reality paper
		int best_g; // the index of the best vanishing point: 0, 1, or 2.
		std::vector<glm::vec2> points;

	public:
		SketchEdge(int type, const std::vector<glm::vec2>& points);
	};

	typedef boost::shared_ptr<SketchEdge> EdgePtr;
}

