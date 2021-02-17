// Calculate a pole of inaccessibility from a nonconvex polygon
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

// Based on implementation by Mapbox (ISC license):
// https://github.com/mapbox/polylabel

#include "pole_of_inaccessibility.h"

namespace arc_perception
{

static double getSegDistSq(const cv::Point2d& p, const cv::Point2d& a, const cv::Point2d& b)
{
	auto x = a.x;
	auto y = a.y;
	auto dx = b.x - x;
	auto dy = b.y - y;

	if (dx != 0 || dy != 0)
	{
		auto t = ((p.x - x) * dx + (p.y - y) * dy) / (dx * dx + dy * dy);

		if (t > 1)
		{
			x = b.x;
			y = b.y;
		}
		else if(t > 0)
		{
			x += dx * t;
			y += dy * t;
		}
	}

	dx = p.x - x;
	dy = p.y - y;

	return dx * dx + dy * dy;
}

// signed distance from point to polygon outline (negative if point is outside)
static double pointToPolygonDist(const cv::Point2d& point, const std::vector<std::vector<cv::Point>>& polygon)
{
	bool inside = false;
	auto minDistSq = std::numeric_limits<double>::infinity();

	for(auto& ring : polygon)
	{
		for (std::size_t i = 0, len = ring.size(), j = len - 1; i < len; j = i++)
		{
			cv::Point2d b = ring[j];
			cv::Point2d a = ring[i];

			if ((a.y > point.y) != (b.y > point.y) &&
				(point.x < (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x)) inside = !inside;

			minDistSq = std::min(minDistSq, getSegDistSq(point, a, b));
		}
	}

    return (inside ? 1 : -1) * std::sqrt(minDistSq);
}

namespace
{
	struct Cell
	{
		Cell(const cv::Point2d& c_, double h_, const std::vector<std::vector<cv::Point>>& polygon)
		 : c(c_)
		 , h(h_)
		 , d(pointToPolygonDist(c, polygon))
		 , max(d + h * std::sqrt(2))
		{}

		cv::Point2d c; // cell center
		double h; // half the cell size
		double d; // distance from cell center to polygon
		double max; // max distance to polygon within a cell
	};
}

// get polygon centroid
static Cell getCentroidCell(const std::vector<std::vector<cv::Point>>& polygon)
{
	double area = 0;
	cv::Point2d c { 0, 0 };

	auto& ring = polygon.at(0);

	for (std::size_t i = 0, len = ring.size(), j = len - 1; i < len; j = i++)
	{
		cv::Point2d a = ring[i];
		cv::Point2d b = ring[j];
		auto f = a.x * b.y - b.x * a.y;
		c.x += (a.x + b.x) * f;
		c.y += (a.y + b.y) * f;
		area += f * 3;
	}

	return Cell(area == 0 ? cv::Point2d(ring.at(0)) : c / area, 0, polygon);
}

cv::Point2d poleOfInaccessibility(const std::vector<std::vector<cv::Point>>& polygon, double precision)
{
	// find the bounding box of the outer ring
	cv::Rect2d envelope = cv::boundingRect(polygon.at(0));
	auto size = envelope.size();

	const double cellSize = std::min(size.width, size.height);
	double h = cellSize / 2;

	// a priority queue of cells in order of their "potential" (max distance to polygon)
	auto compareMax = [] (const Cell& a, const Cell& b) {
		return a.max < b.max;
	};

	using Queue = std::priority_queue<Cell, std::vector<Cell>, decltype(compareMax)>;
	Queue cellQueue(compareMax);

	if (cellSize == 0) {
		return envelope.tl();
	}

	// cover polygon with initial cells
	for (double x = envelope.x; x < envelope.x + envelope.width; x += cellSize) {
		for (double y = envelope.y; y < envelope.y + envelope.height; y += cellSize) {
			cellQueue.push(Cell({x + h, y + h}, h, polygon));
		}
	}

	// take centroid as the first best guess
	auto bestCell = getCentroidCell(polygon);

	// special case for rectangular polygons
	Cell bboxCell(cv::Point2d(envelope.tl()) + cv::Point2d(size.width, size.height) / 2.0, 0, polygon);
	if (bboxCell.d > bestCell.d) {
		bestCell = bboxCell;
	}

	auto numProbes = cellQueue.size();
	while (!cellQueue.empty()) {
		// pick the most promising cell from the queue
		auto cell = cellQueue.top();
		cellQueue.pop();

		// update the best cell if we found a better one
		if (cell.d > bestCell.d) {
			bestCell = cell;
// 			if (debug) std::cout << "found best " << std::round(1e4 * cell.d) / 1e4 << " after " << numProbes << " probes" << std::endl;
		}

		// do not drill down further if there's no chance of a better solution
		if (cell.max - bestCell.d <= precision) continue;

		// split the cell into four cells
		h = cell.h / 2;
		cellQueue.push(Cell({cell.c.x - h, cell.c.y - h}, h, polygon));
		cellQueue.push(Cell({cell.c.x + h, cell.c.y - h}, h, polygon));
		cellQueue.push(Cell({cell.c.x - h, cell.c.y + h}, h, polygon));
		cellQueue.push(Cell({cell.c.x + h, cell.c.y + h}, h, polygon));
		numProbes += 4;
	}

// 	if (debug) {
// 		std::cout << "num probes: " << numProbes << std::endl;
// 		std::cout << "best distance: " << bestCell.d << std::endl;
// 	}

	return bestCell.c;
}

cv::Point2d polygonCentroid(const std::vector<cv::Point>& polygon)
{
	double A = 0.0;
	cv::Point2d centroid(0,0);

	for(std::size_t i = 0; i < polygon.size() - 1; ++i)
	{
		double a = polygon[i].x * polygon[i+1].y - polygon[i+1].x * polygon[i].y;

		A += a;
		centroid.x += (polygon[i].x + polygon[i+1].x) * a;
		centroid.y += (polygon[i].y + polygon[i+1].y) * a;
	}

	// Close the loop
	{
		double a = polygon.back().x * polygon[0].y - polygon[0].x * polygon.back().y;

		A += a;
		centroid.x += (polygon.back().x + polygon[0].x) * a;
		centroid.y += (polygon.back().y + polygon[0].y) * a;
	}

	A /= 2;
	centroid.x /= 6.0 * A;
	centroid.y /= 6.0 * A;

	return centroid;
}

double contourDistance(const std::vector<std::vector<cv::Point> >& polygon, const cv::Point2d& point)
{
	return pointToPolygonDist(point, polygon);
}

}
