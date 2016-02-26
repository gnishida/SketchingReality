#include "SketchEdge.h"

namespace sketch {

	SketchEdge::SketchEdge(int type, const std::vector<glm::vec2>& points) {
		this->type = type;
		this->points = points;
		this->g_values.resize(3, 0);
	}
}