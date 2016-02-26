#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <vector>

namespace utils {

	float uniform();
	float uniform(float a, float b);
	float pointSegmentDistance(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c, glm::vec2& closestPtInAB);
	bool segmentSegmentIntersect(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c, const glm::vec2& d, float *tab, float *tcd, bool segmentOnly, glm::vec2& intPoint);
	float area(const std::vector<glm::vec2>& points);

}