#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <boost/shared_ptr.hpp>

namespace sketch {

	class SketchVertex {
	public:
		static enum { TYPE_ISOLATED = 0, TYPE_LINE, TYPE_L, TYPE_T, TYPE_Y, TYPE_E, TYPE_X };

	public:
		int type;
		glm::vec2 pt;

	public:
		SketchVertex(const glm::vec2& pt);
	};

	typedef boost::shared_ptr<SketchVertex> VertexPtr;
}

