// Weighted directed graph
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "graph.h"

namespace graph
{

Graph::Graph(unsigned int numVertices)
 : m_vertices(numVertices)
{
}

Graph::~Graph()
{
}

void Graph::addEdge(VertexID from, VertexID to, double weight)
{
	m_vertices[from].successors.push_back(m_edges.size());
	m_vertices[to].predecessors.push_back(m_edges.size());
	m_edges.emplace_back(from, to, weight);
}

void Graph::reconstructSuccessors()
{
	for(auto& v : m_vertices)
	{
		v.successors.clear();
		v.predecessors.clear();
	}

	for(std::size_t edgeID = 0; edgeID < m_edges.size(); ++edgeID)
	{
		m_vertices[m_edges[edgeID].from].successors.push_back(edgeID);
		m_vertices[m_edges[edgeID].to].predecessors.push_back(edgeID);
	}
}

void Graph::toDot(std::ostream& stream, const std::vector<std::string>& labels, const std::function<bool(VertexID)>& vertexPredicate)
{
	stream << "strict digraph test {\n";

	for(graph::VertexID v = 0; v < m_vertices.size(); ++v)
	{
		if(vertexPredicate(v))
			stream << "n" << v << " [label=\"" << labels[v] << "\"];\n";
	}

	for(const auto& edge : m_edges)
	{
		if(!vertexPredicate(edge.from) || !vertexPredicate(edge.to))
			continue;

		stream << "n" << edge.from << " -> n" << edge.to << " [label=\"" << edge.weight << "\"];\n";
	}
	stream << "}\n";
}

}
