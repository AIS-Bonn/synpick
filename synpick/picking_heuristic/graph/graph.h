// Weighted directed graph
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <iostream>
#include <functional>

namespace graph
{

typedef unsigned int EdgeID;
typedef unsigned int VertexID;

constexpr VertexID UNSET = -1;

struct Vertex
{
	std::vector<EdgeID> predecessors;
	std::vector<EdgeID> successors;
};

struct Edge
{
	Edge(VertexID from, VertexID to, double weight)
	 : from(from)
	 , to(to)
	 , weight(weight)
	{}

	VertexID from;
	VertexID to;
	double weight;
};

class Graph
{
public:
	Graph(unsigned int numVertices);
	~Graph();

	void addEdge(VertexID from, VertexID to, double weight);

	inline std::vector<Edge>& edges()
	{ return m_edges; }

	inline const std::vector<Edge>& edges() const
	{ return m_edges; }

	inline std::vector<Vertex>& vertices()
	{ return m_vertices; }

	inline const std::vector<Vertex>& vertices() const
	{ return m_vertices; }

	void reconstructSuccessors();

	void toDot(std::ostream& stream, const std::vector<std::string>& labels, const std::function<bool(VertexID)>& vertexPredicate = [](VertexID){return true;});
private:
	std::vector<Vertex> m_vertices;
	std::vector<Edge> m_edges;
};

}

#endif

