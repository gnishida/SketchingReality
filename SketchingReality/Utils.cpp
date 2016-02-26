#include "Utils.h"
#include <random>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/planar_face_traversal.hpp>
#include <boost/graph/boyer_myrvold_planar_test.hpp>
#include <boost/geometry/geometries/polygon.hpp> 
#include <boost/geometry/geometries/point_xy.hpp> 
#include <boost/geometry.hpp>
#ifndef SQR
#define SQR(x)	((x)*(x))
#endif

namespace utils {
	
	const float FLOAT_TOL = 1e-6f;

	float uniform() {
		return (float)(rand() % 100) / 100.0f;
	}

	float uniform(float a, float b) {
		return uniform() * (b - a) + a;
	}

	float pointSegmentDistance(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c, glm::vec2& closestPtInAB) {
		float dist;

		float r_numerator = (c.x - a.x)*(b.x - a.x) + (c.y - a.y)*(b.y - a.y);
		float r_denomenator = (b.x - a.x)*(b.x - a.x) + (b.y - a.y)*(b.y - a.y);

		// For the case that the denominator is 0.
		if (r_denomenator <= 0.0f) {
			closestPtInAB = a;
			return (a - c).length();
		}

		float r = r_numerator / r_denomenator;

		//
		float px = a.x + r*(b.x - a.x);
		float py = a.y + r*(b.y - a.y);
		//    
		float s = ((a.y - c.y)*(b.x - a.x) - (a.x - c.x)*(b.y - a.y)) / r_denomenator;

		float distanceLine = fabs(s)*sqrt(r_denomenator);

		closestPtInAB = glm::vec2(px, py);

		if ((r >= 0) && (r <= 1)) {
			dist = distanceLine;
		}
		else {
			float dist1 = (c.x - a.x)*(c.x - a.x) + (c.y - a.y)*(c.y - a.y);
			float dist2 = (c.x - b.x)*(c.x - b.x) + (c.y - b.y)*(c.y - b.y);
			if (dist1 < dist2) {
				dist = sqrt(dist1);
				closestPtInAB = a;
			}
			else {
				dist = sqrt(dist2);
				closestPtInAB = b;
			}
		}

		return abs(dist);
	}

	/**
	* Computes the intersection between two line segments on the XY plane.
	* Segments must intersect within their extents for the intersection to be valid. z coordinate is ignored.
	*
	* @param a one end of the first line
	* @param b another end of the first line
	* @param c one end of the second line
	* @param d another end of the second line
	* @param tab
	* @param tcd
	* @param segmentOnly
	* @param intPoint	the intersection
	* @return true if two lines intersect / false otherwise
	**/
	bool segmentSegmentIntersect(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c, const glm::vec2& d, float *tab, float *tcd, bool segmentOnly, glm::vec2& intPoint) {
		glm::vec2 u = b - a;
		glm::vec2 v = d - c;

		if (glm::length(u) < FLOAT_TOL || glm::length(u) < FLOAT_TOL) {
			return false;
		}

		float numer = v.x*(c.y - a.y) + v.y*(a.x - c.x);
		float denom = u.y*v.x - u.x*v.y;

		if (denom == 0.0f)  {
			// they are parallel
			*tab = 0.0f;
			*tcd = 0.0f;
			return false;
		}

		float t0 = numer / denom;

		glm::vec2 ipt = a + t0*u;
		glm::vec2 tmp = ipt - c;
		float t1;
		if (glm::dot(tmp, v) > 0.0f) {
			t1 = tmp.length() / v.length();
		}
		else {
			t1 = -1.0f * tmp.length() / v.length();
		}

		//Check if intersection is within segments
		if (segmentOnly && !((t0 >= FLOAT_TOL) && (t0 <= 1.0f - FLOAT_TOL) && (t1 >= FLOAT_TOL) && (t1 <= 1.0f - FLOAT_TOL))){
			return false;
		}

		*tab = t0;
		*tcd = t1;
		glm::vec2 dirVec = b - a;

		intPoint = a + (*tab)*dirVec;

		return true;
	}

	float area(const std::vector<glm::vec2>& points) {
		boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double> > polygon;
		for (int i = 0; i < points.size(); ++i) {
			polygon.outer().push_back(boost::geometry::model::d2::point_xy<double>(points[i].x, points[i].y));
		}
		return -boost::geometry::area(polygon);
	}
}